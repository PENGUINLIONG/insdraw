use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::iter::repeat;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::c_char;
use log::{info, warn, error, debug, trace};
use ash::vk;
use ash::{vk_make_version, Entry, Instance, Device};
use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use spirq::SpirvBinary;
use spirq::reflect::{EntryPoint as EntryPointManifest, Pipeline as PipelineManifest};
use super::error::{Error, Result};

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
    api_exts: Vec<&'static CStr>,
    dev_exts: Vec<&'static CStr>,
    feats: vk::PhysicalDeviceFeatures,
}
impl ContextBuilder {
    const API_VERSION: u32 = vk_make_version!(1, 1, 0);
    pub fn filter_device<DevSel: 'static + Fn(&vk::PhysicalDeviceProperties) -> bool>(mut self, dev_sel: DevSel) -> Self {
        self.dev_sel = Some(Box::new(dev_sel));
        self
    }
    pub fn with_api_extensions(&mut self, api_exts: &[&'static CStr]) -> &mut Self {
        self.api_exts = api_exts.to_owned();
        self
    }
    pub fn with_device_extensions(&mut self, dev_exts: &[&'static CStr]) -> &mut Self {
        self.dev_exts = dev_exts.to_owned();
        self
    }
    pub fn with_interface(&mut self, icfg: InterfaceConfig) -> &mut Self {
        // Sort queue configs by flag complexity, in descending order. Complex
        // requirements should be met first.
        self.icfgs.push(icfg);
        self
    }
    pub fn with_features(&mut self, feats: vk::PhysicalDeviceFeatures) -> &mut Self {
        self.feats = feats;
        self
    }
    fn make_ext_prop(ext_name: &str) -> vk::ExtensionProperties {
        let mut props = vk::ExtensionProperties {
            extension_name: [0; 256],
            spec_version: Self::API_VERSION,
        };
        let ext_name_len = ext_name.len();
        // Do not check equal because '\0' take a byte.
        assert!(ext_name_len < vk::MAX_EXTENSION_NAME_SIZE,
            "extension name too long");
        let ext_name = unsafe { std::slice::from_raw_parts(ext_name.as_ptr() as *const c_char, ext_name_len) };
        props.extension_name[..ext_name.len()].copy_from_slice(ext_name);
        return props;
    }
    //make_ext_prop("VK_KHR_surface", vk_make_version!(1, 0, 0));
    fn filter_api_exts(&self, entry: &Entry) -> Result<Vec<*const c_char>> {
        let layer_name = std::ptr::null_mut::<c_char>();
        let ext_props = unsafe { entry.enumerate_instance_extension_properties() }?;
        let avail_exts = ext_props.iter()
            .filter_map(|ext_prop| {
                if ext_prop.spec_version <= Self::API_VERSION {
                    unsafe { Some(CStr::from_ptr(ext_prop.extension_name.as_ptr())) }
                } else { None }
            })
            .collect::<HashSet<_>>();
        trace!("instance supported extensions: {:?}", avail_exts);
        let exts = self.api_exts.iter()
            .filter_map(|x| {
                if avail_exts.contains(x) {
                    Some(x.as_ptr())
                } else {
                    warn!("instance doesn't support extension '{}'", x.to_string_lossy());
                    None
                }
            })
            .collect::<Vec<_>>();
        return Ok(exts);
    }
    fn filter_dev_exts(&self, inst: &Instance, physdev: vk::PhysicalDevice) -> Result<Vec<*const c_char>> {
        let layer_name = std::ptr::null_mut::<c_char>();
        let ext_props = unsafe { inst.enumerate_device_extension_properties(physdev) }?;
        let avail_exts = ext_props.iter()
            .filter_map(|ext_prop| {
                if ext_prop.spec_version <= Self::API_VERSION {
                    unsafe { Some(CStr::from_ptr(ext_prop.extension_name.as_ptr())) }
                } else { None }
            })
            .collect::<HashSet<_>>();
        trace!("device supported extensions: {:?}", avail_exts);
        let exts = self.dev_exts.iter()
            .filter_map(|x| {
                if avail_exts.contains(x) {
                    Some(x.as_ptr())
                } else {
                    warn!("device doesn't support extension '{}'", x.to_string_lossy());
                    None
                }
            })
            .collect::<Vec<_>>();
        return Ok(exts);
    }
    fn try_create_inst(&self, entry: &Entry) -> Result<Instance> {
        // Create vulkan instance.
        let app_name = CString::new(self.app_name).unwrap();
        let engine_name = CString::new("insdraw").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .api_version(Self::API_VERSION)
            .application_name(&app_name)
            .application_version(vk_make_version!(0, 0, 1))
            .engine_name(&engine_name)
            .engine_version(vk_make_version!(0, 0, 1))
            .build();
        let api_exts = self.filter_api_exts(&entry)?;
        let inst_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&api_exts)
            .build();
        let inst = unsafe { entry.create_instance(&inst_create_info, None)? };
        Ok(inst)
    }
    fn try_create_dev(&mut self, inst: &Instance, physdev: vk::PhysicalDevice) -> Result<(Device, HashMap<&'static str, vk::Queue>)> {
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
            return Err(Error::NoCapablePhysicalDevice);
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
        let dev_exts = self.filter_dev_exts(inst, physdev)?;
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
    pub fn build(&mut self) -> Result<Context> {
        let entry = Entry::new()?;
        let inst = self.try_create_inst(&entry)?;
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
            .ok_or(Error::NoCapablePhysicalDevice)?;
        let ctxt = Context {
            entry: entry,
            inst: inst,
            dev: dev,
            queues: queues,
            api_exts: Default::default(),
            dev_exts: Default::default(),
        };
        Ok(ctxt)
    }
}


pub struct Context {
    // Don't change the order. Things should be dropped from top to bottom.
    pub queues: HashMap<&'static str, vk::Queue>,
    pub dev: Device,
    pub inst: Instance,
    pub entry: Entry,
    pub api_exts: HashSet<&'static CStr>,
    pub dev_exts: HashSet<&'static CStr>,
}
impl Context {
    pub fn builder(app_name: &'static str) -> ContextBuilder {
        ContextBuilder {
            app_name: app_name,
            ..Default::default()
        }
    }
}
impl PartialEq for Context {
    fn eq(&self, rhs: &Self) -> bool { self.dev.handle() == rhs.dev.handle() }
}
impl Eq for Context {}


pub struct ShaderModule<'a> {
    ctxt: &'a Context,
    pub handle: vk::ShaderModule,
    entry_points: HashMap<String, EntryPointManifest>,
}
impl<'a> ShaderModule<'a> {
    pub fn new(ctxt: &'a Context, spv: &SpirvBinary) -> Result<ShaderModule<'a>> {
        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(spv.words())
            .build();
        let handle = unsafe { ctxt.dev.create_shader_module(&create_info, None)? };
        let entry_points = spv.reflect()?
            .into_iter()
            .map(|x| (x.name.to_owned(), x.clone()))
            .collect::<HashMap<_, _>>();
        let shader_mod = ShaderModule {
            ctxt: ctxt,
            handle: handle,
            entry_points: entry_points,
        };
        info!("created shader module");
        Ok(shader_mod)
    }
    pub fn entry_points(&self) -> impl Iterator<Item=&EntryPointManifest> {
        self.entry_points.values()
    }
    pub fn get(&self, entry_point_name: &str) -> Option<&EntryPointManifest> {
        self.entry_points.get(entry_point_name)
    }
}
impl<'a> Drop for ShaderModule<'a> {
    fn drop(&mut self) {
        unsafe { self.ctxt.dev.destroy_shader_module(self.handle, None) };
        info!("destroyed shader module");
    }
}

/*
pub struct RenderPass {
    handle: vk::RenderPass,
    subpasses: Vec<Subpass>,
}
pub struct Subpass {
    handle: vk::Pipeline,
    manifest: Manifest,
    vert_binds: VertexBinding,
}


#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct VertexBinding {
    bind_point: u32,
    stride: usize,
    input_rate: vk::VertexInputRate,
}
impl VertexBinding {
    pub fn per_vertex(bind_point: u32, stride: usize) -> VertexBinding {
        VertexBinding {
            bind_point: bind_point,
            stride: stride,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }
    pub fn per_instance(bind_point: u32, stride: usize) -> VertexBinding {
        VertexBinding {
            bind_point: bind_point,
            stride: stride,
            input_rate: vk::VertexInputRate::INSTANCE,
        }
    }
}


#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct VertexAttribute {
    vert_bind: &'_ VertexBinding,
    location: Location,
    offset: usize,
    fmt: vk::Format,
}
impl VertexAttribute {
    pub fn new(vert_bind: &VertexBinding, location: Location, offset: usize, fmt: vk::Format) -> VertexAttribute {
        VertexAttribute {
            vert_bind: vert_bind,
            location: location,
            offset: offset,
            fmt: fmt,
        }
    }
}


pub struct Pipeline {
    vert_attr
    entry_points: Vec<EntryPointManifest>,
}
impl Pipeline {
    pub fn new(entry_points: Vec<EntryPointManifest>) -> Pipeline {

    }
}

pub struct Attachment {
    location: Location,
    load_op: vk::AttachmentLoadOp,
    store_op: vk::AttachmentStoreOp,
    stencil_load_op: vk::AttachmentLoadOp,
    stencil_store_op: vk::AttachmentStoreOp,
    ms_rate: vk::SampleCountFlagBits,
    fmt: vk::Format,
    from_layout: vk::ImageLayout,
    to_layout: vk::ImageLayout,
}

struct AttributeMapping {
    vert_attrs: HashMap<Location, VertexAttribute>;
    vert_binds: HashMap<BindingPoint, VertexBinding>;
}
*/
