#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum State {
    Vacant,
    Split,
    Occupied,
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
    // Allocate a piece of memory in the managed memory page. Returns the
    // address (offset from the beginning of the memory pate) to the allocated
    // memory.
    pub fn alloc(&mut self, alloc_size: usize) -> Option<usize> {
        // Reject invalid sizes right away.
        if alloc_size == 0 || alloc_size > self.size { return None }
        const NBIT_USIZE: u32 = (std::mem::size_of::<usize>() * 8) as u32;
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
                i += 1;
            } else if let Some(occupied_rorder) = (0..rorder).into_iter().rev()
                .filter_map(|cur_rorder| {
                    let cur_order = self.max_order - cur_rorder;
                    match self.get_state(i >> cur_order, cur_rorder) {
                        State::Occupied => Some(cur_rorder),
                        State::Vacant => { top_vacant_rorder = cur_rorder; None },
                        _ => None,
                    }
                })
                .next() {
                // Bubble up and check if any of the parent block has already
                // been occupied. If so, skip the entire parent block.
                let delta_rorder = rorder - occupied_rorder;
                i >>= delta_rorder;
                i += 1;
                i <<= delta_rorder;
            } else {
                // Found a vacant block. Occupy the block and split higher level
                // blocks.
                self.buddies[order_base + i] = State::Occupied;
                for cur_rorder in top_vacant_rorder..rorder {
                    let cur_order = self.max_order - cur_rorder;
                    self.set_state(i >> cur_order, cur_rorder, State::Split);
                }
                let offset = i * (self.unit << nbit_addr);
                println!("{:?}", self.buddies);
                return Some(offset);
            }
        }
        None
    }
    // Release a piece of allocated memory located at `addr`. If the memory
    // address has not been allocated yet, the method works as an no-op.
    pub fn free(&mut self, addr: usize) -> Option<()>{
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
                State::Vacant => return Some(()),
                // Discovered the actual order the memory has been allocated.
                State::Occupied => { nbit_addr = cur_order; break },
                _ => {},
            }
        }

        // Release allocation and merge parents (if possible).
        self.set_state(unit_idx >> nbit_addr, rorder, State::Vacant);
        for cur_rorder in 0..rorder {
            // We only have to check the neighbors' state because the states of
            // parent-blocks are managed by us. Here we toggle the LSB to switch
            // to the other bisection.
            let cur_child_rorder = cur_rorder + 1;
            let cur_child_order = self.max_order - cur_child_rorder;
            let child_i = (unit_idx >> cur_child_order) ^ 1;
            if self.get_state(child_i, cur_child_order) != State::Vacant { break }
            let cur_order = self.max_order - cur_rorder;
            let i = unit_idx >> cur_order;
            self.set_state(i, cur_rorder, State::Vacant);
        }
        println!("{:?}", self.buddies);
        Some(())
    }
}
