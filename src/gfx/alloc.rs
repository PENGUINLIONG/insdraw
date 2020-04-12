use log::trace;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum State {
    Vacant,
    Split,
    Occupied(usize),
}

pub struct BuddyAllocator {
    size: usize,
    unit: usize,
    max_order: u32,
    buddies: Vec<State>,
}
impl BuddyAllocator {
    pub fn new(size: usize, max_order: u32) -> BuddyAllocator {
        let unit = size >> max_order;
        assert!(unit != 0, "too much order");
        let nelem = (1 << (max_order + 1)) - 1;
        BuddyAllocator {
            size: size,
            unit: unit,
            max_order: max_order,
            buddies: (0..nelem).into_iter()
                .map(|_| State::Vacant)
                .collect::<Vec<_>>(),
        }
    }
    pub fn is_unused(&self) -> bool {
        self.get_state(0, 0) == State::Vacant
    }

    #[inline]
    fn get_state(&self, i: usize, rorder: u32) -> State {
        let len = 1 << rorder;
        debug_assert!(i < len);
        let base = len - 1;
        self.buddies[base + i]
    }
    #[inline]
    fn set_state(&mut self, i: usize, rorder: u32, state: State) {
        let len = 1 << rorder;
        debug_assert!(i < len);
        let base = len - 1;
        self.buddies[base + i] = state;
    }
    /// Allocate a piece of memory in the managed memory page. Returns the
    /// address (offset from the beginning of the memory pate) to the allocated
    /// memory.
    ///
    /// WARN: It should be noted that the reference count is initially set to 0.
    /// The user code MUST `refer` to the base address of the allocated memory
    /// otherwise the mechanism will break.
    pub fn alloc(&mut self, alloc_size: usize, align: usize) -> Option<usize> {
        const NBIT_USIZE: u32 = (std::mem::size_of::<usize>() * 8) as u32;
        // Reject invalid sizes right away.
        if alloc_size == 0 || alloc_size > self.size { return None }
        debug_assert!(!align.is_power_of_two(),
            "doesn't support non pow-of-2 alignment");
        // Step of each iteration.
        let step = (align + self.unit - 1) / self.unit;
        // The number of units needed to contain the allocation.
        let nunit = (alloc_size + self.unit - 1) / self.unit;
        // The length of integer needed to describe the allocation size. With
        // this number of bits the user can address any unit within the
        // allocation. This is also the order number of the allocation.
        let nbit_addr = NBIT_USIZE - (nunit - 1).leading_zeros();
        // Calculate the reversed order number, which is:
        let rorder = self.max_order - nbit_addr;

        // Number of blocks in the order.
        let nblock = 1 << rorder;
        // Base index to the order.
        let order_base = nblock - 1;

        // Index to the current working block within the order.
        let mut i = 0;
        while i < nblock {
            // The order where the topmost vacant block in the working block's
            // parent chain is located.
            let mut top_vacant_rorder = 0;
            if self.buddies[order_base + i] != State::Vacant {
                // The block is not vacant or has been split in to sub-blocks.
                // Don't care.
                i += step;
            } else if let Some(occupied_rorder) = (0..rorder).into_iter().rev()
                .filter_map(|cur_rorder| {
                    let cur_order = self.max_order - cur_rorder;
                    match self.get_state(i >> cur_order, cur_rorder) {
                        State::Occupied(_) => Some(cur_rorder),
                        State::Vacant => { top_vacant_rorder = cur_rorder; None },
                        _ => None,
                    }
                })
                .next() {
                // Bubble up and check if any of the parent block has already
                // been occupied. If so, skip the entire parent block.
                let delta_rorder = rorder - occupied_rorder;
                i >>= delta_rorder;
                i += step;
                i <<= delta_rorder;
            } else {
                // Found a vacant block. Occupy the block and split higher level
                // blocks.
                self.buddies[order_base + i] = State::Occupied(0);
                for cur_rorder in top_vacant_rorder..rorder {
                    let cur_order = self.max_order - cur_rorder;
                    self.set_state(i >> cur_order, cur_rorder, State::Split);
                }
                let offset = i * (self.unit << nbit_addr);
                trace!("allocated {} bytes at offset {:#x}", alloc_size, offset);
                return Some(offset);
            }
        }
        None
    }
    fn get_addr_occupied_location(&self, addr: usize) -> Option<(usize, u32)> {
        // Reject unaddressable locations right away.
        if addr >= self.size || addr % self.unit != 0 { return None }
        let unit_idx = addr / self.unit;
        // Order number and reversed order number will be updated with actual
        // values.
        let mut nbit_addr = unit_idx.trailing_zeros().min(self.max_order);
        let mut rorder = self.max_order - nbit_addr;

        // Try getting the actual allocation order number.
        for cur_order in (0..=nbit_addr).into_iter().rev() {
            let i = unit_idx >> cur_order;
            rorder = self.max_order - cur_order;
            match self.get_state(i, rorder) {
                // This address hasn't been allocated yet. Cast no side-effect.
                State::Vacant => return None,
                // Discovered the actual order the memory has been allocated.
                State::Occupied(_) => return Some((i, rorder)),
                _ => {},
            }
        }
        None
    }
    // Release a piece of allocated memory located at `addr`. If the memory
    // address has not been allocated yet, the method works as an no-op. `true`
    // is returned when the memory needs to be freed since the reference count
    // is reduced to 0.
    pub fn free(&mut self, addr: usize) -> Option<bool> {
        let (i, rorder) = self.get_addr_occupied_location(addr)?;
        match self.get_state(i, rorder) {
            // Discovered the actual order the memory has been allocated.
            State::Occupied(ref_count) => {
                // Reduce reference count.
                self.set_state(i, rorder, State::Occupied(ref_count - 1));
                trace!("decreased reference counting at offset {:#x}", addr);
                Some(false)
            },
            State::Occupied(1) => {
                // After this free the reference count is decreased to 0.
                // There will be no referrer, so release allocation and merge
                // parents (if possible).
                self.set_state(i, rorder, State::Vacant);
                let unit_idx = addr / self.unit;
                for cur_rorder in 0..rorder {
                    // We only have to check the neighbors' state because the
                    // states of parent-blocks are managed by us. Here we toggle
                    // the LSB to switch to the other bisection.
                    let cur_child_rorder = cur_rorder + 1;
                    let cur_child_order = self.max_order - cur_child_rorder;
                    let child_i = (unit_idx >> cur_child_order) ^ 1;
                    let cur_state = self.get_state(child_i, cur_child_order);
                    if cur_state != State::Vacant { break }
                    let cur_order = self.max_order - cur_rorder;
                    let i = unit_idx >> cur_order;
                    trace!("freed memory at offset {:#x}", addr);
                    self.set_state(i, cur_rorder, State::Vacant);
                }
                Some(true)
            },
            _ => unreachable!(),
        }
    }
    // Increase the reference count at `addr`. This method works as an no-op if
    // the address hasn't been allocated yet.
    pub fn refer(&mut self, addr: usize) -> Option<()> {
        let (i, rorder) = self.get_addr_occupied_location(addr)?;
        match self.get_state(i, rorder) {
            State::Occupied(ref_count) => {
                self.set_state(i, rorder, State::Occupied(ref_count + 1));
                trace!("increased reference counting at offset {:#x}", addr);
                Some(())
            },
            _ => unreachable!(),
        }
    }
}
