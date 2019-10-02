use std::cmp::Reverse;
use std::collections::HashMap;
use std::iter::repeat;
use std::ffi::{CStr, CString};
use log::{info};
use ash::vk;
use ash::{vk_make_version, Entry, Instance, Device};
use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use super::GraphicsError;

#[derive(Default)]
pub struct InterfaceConfig {
    name: &'static str,
    flags: vk::QueueFlags,
    priority: f32,
}
impl InterfaceConfig {
    pub fn new(name: &'static str) -> Self {
        InterfaceConfig {
            name: name,
            ..Default::default()
        }
    }
    pub fn with_priority(mut self, priority: f32) -> Self {
        self.priority = priority;
        self
    }
    pub fn require_transfer(mut self) -> Self {
        self.flags |= vk::QueueFlags::TRANSFER;
        self
    }
    pub fn require_graphics(mut self) -> Self {
        self.flags |= vk::QueueFlags::GRAPHICS;
        self
    }
    pub fn require_compute(mut self) -> Self {
        self.flags |= vk::QueueFlags::COMPUTE;
        self
    }
}

#[derive(Default)]
pub struct ContextBuilder {
    app_name: &'static str,
    icfgs: Vec<InterfaceConfig>,
    dev_sel: Option<Box<dyn Fn(&vk::PhysicalDeviceProperties) -> bool>>,
    api_exts: Vec<CString>,
    dev_exts: Vec<CString>,
    feats: vk::PhysicalDeviceFeatures,
}
impl ContextBuilder {
    pub fn filter_device<DevSel: 'static + Fn(&vk::PhysicalDeviceProperties) -> bool>(mut self, dev_sel: DevSel) -> Self {
        self.dev_sel = Some(Box::new(dev_sel));
        self
    }
    pub fn with_api_extensions(&mut self, api_exts: &'static [&'static str]) -> Self {
        self.api_exts = api_exts.iter()
            .map(|&x| CString::new(x).unwrap())
            .collect::<Vec<_>>();
        self
    }
    pub fn with_device_extensions(&mut self, dev_exts: &'static [&'static str]) -> Self {
        self.dev_exts = dev_exts.iter()
            .map(|&x| CString::new(x).unwrap())
            .collect::<Vec<_>>();
        self
    }
    pub fn with_interface(&mut self, icfg: InterfaceConfig) -> Self {
        // Sort queue configs by flag complexity, in descending order. Complex
        // requirements should be met first.
        self.icfgs.push(icfg);
        self
    }
    pub fn with_features(&mut self, feats: vk::PhysicalDeviceFeatures) -> Self {
        self.feats = feats;
        self
    }
    fn try_create_inst(&self) -> GraphicsResult<(Entry, Instance)> {
        use ash::extensions::{khr::Surface, khr::Win32Surface};
        let entry = Entry::new()?;
        // Create vulkan instance.
        let app_name = CString::new(self.app_name).unwrap();
        let engine_name = CString::new("insdraw").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .api_version(vk_make_version!(1, 1, 0))
            .application_name(&app_name)
            .application_version(vk_make_version!(0, 0, 1))
            .engine_name(&engine_name)
            .engine_version(vk_make_version!(0, 0, 1))
            .build();
        let api_exts = self.api_exts.iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();
        let inst_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&api_exts)
            .build();
        let inst = unsafe { entry.create_instance(&inst_create_info, None)? };
        Ok((entry, inst))
    }
    fn try_create_dev(&mut self, inst: &Instance, physdev: vk::PhysicalDevice) -> GraphicsResult<(Device, HashMap<&'static str, vk::Queue>)> {
        // In following codes, `i` is queue family index; `j` is queue index in
        // a family; and `k` is the index of interface.

        // Assign physical device queues.
        let qfam_props = unsafe { inst.get_physical_device_queue_family_properties(physdev) };
        let mut qfam_caps = qfam_props.iter()
            .map(|x| x.queue_flags)
            .enumerate()
            .collect::<Vec<_>>();
        // Sort available queues by flag complexity in ascending order.
        // Versatile queues should be kept for more complicated requirements.
        qfam_caps.sort_by_key(|&(_i, flags)| flags.as_raw().count_ones());
        let qfam_counts = qfam_props.iter()
            .map(|x| x.queue_count)
            .collect::<Vec<_>>();
        let mut qfam_queue_idxs = repeat(0)
            .take(qfam_counts.len())
            .collect::<Vec<_>>();
        // Sort available queues by flag complexity in descending order. Complex
        // requirements should be met first.
        self.icfgs.sort_by_key(|x| Reverse(x.flags.as_raw().count_ones()));
        let interface_qfam_idxs = self.icfgs.iter()
            .enumerate()
            .filter_map(|(k, icfg)| {
                // Find a simplest queue that meets the requirement of current
                // interface config.
                qfam_caps.iter()
                    .find_map(|&(i, flags)| {
                        let j = qfam_queue_idxs[i];
                        if flags.contains(icfg.flags) && qfam_counts[i] > j {
                            qfam_queue_idxs[i] += 1;
                            Some((i, j, k))
                        } else { None }
                    })
            })
            .collect::<Vec<_>>();
        if interface_qfam_idxs.len() != self.icfgs.len() {
            return Err(GraphicsError::NoCapablePhysicalDevice);
        }
        // Mapping from queue family index to queue priorities.
        let mut priorities = HashMap::<usize, Vec<f32>>::new();
        for &(i, _, k) in interface_qfam_idxs.iter() {
            priorities.entry(i).or_default().push(self.icfgs[k].priority);
        }
        let create_infos = priorities.values()
            .enumerate()
            .map(|(i, x)| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(i as u32)
                    .queue_priorities(x)
                    .build()
            })
            .collect::<Vec<_>>();
        // Create device.
        let dev_exts = self.dev_exts.iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();
        let dev_create_info = vk::DeviceCreateInfo::builder()
            .enabled_features(&self.feats)
            .queue_create_infos(&create_infos)
            .enabled_extension_names(&dev_exts)
            .build();
        let dev = unsafe { inst.create_device(physdev, &dev_create_info, None)? };
        // Extract queues.
        let queues = interface_qfam_idxs.into_iter()
            .map(|(i, j, k)| {
                let name = self.icfgs[k].name;
                info!("set up queue for interface '{}'", name);
                let queue = unsafe { dev.get_device_queue(i as u32, j) };
                (name, queue)
            })
            .collect::<HashMap<_, _>>();
        Ok((dev, queues))
    }
    pub fn build(&mut self) -> GraphicsResult<Context> {
        let (entry, inst) = self.try_create_inst()?;
        info!("created vulkan instance");
        let physdevs = unsafe { inst.enumerate_physical_devices() }?;
        info!("discovered {} physical devices", physdevs.len());
        let (dev, queues) = physdevs.into_iter()
            .find_map(|physdev| {
                if let Some(sel) = &self.dev_sel {
                    let prop = unsafe { inst.get_physical_device_properties(physdev) };
                    let dev_name = unsafe { CStr::from_ptr(prop.device_name.as_ptr()).to_string_lossy() };
                    info!("checking '{}' ({:?})...", dev_name, prop.device_type);
                    if sel(&prop) {
                        if let Ok(x) = self.try_create_dev(&inst, physdev) {
                            info!("created device on '{}' ({:?})", dev_name, prop.device_type);
                            return Some(x)
                        }
                    }
                }
                None
            })
            .ok_or(GraphicsError::NoCapablePhysicalDevice)?;
        let ctxt = Context {
            entry: entry,
            inst: inst,
            dev: dev,
            queues: queues,
        };
        Ok(ctxt)
    }
}
pub struct Context {
    // Don't change the order. Things should be dropped from top to bottom.
    queues: HashMap<&str, vk::Queue>,
    dev: Device,
    inst: Instance,
    entry: Entry,
}
impl Context {
    pub fn builder(app_name: &'static str) -> ContextBuilder {
        ContextBuilder {
            app_name: app_name,
            ..Default::default()
        }
    }
}
