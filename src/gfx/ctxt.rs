use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::iter::repeat;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::c_char;
use std::sync::Arc;
use log::{info, warn, error, debug, trace};
use ash::vk;
use ash::vk_make_version;
use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use ash::extensions as vkx;
use lazy_static::lazy_static;
use spirq::SpirvBinary;
use spirq::reflect::{EntryPoint as EntryPointManifest, Pipeline as PipelineManifest};
use winit::window::Window;
use super::error::{Error, Result};

const API_VERSION: u32 = vk_make_version!(1, 1, 0);

lazy_static! {
    static ref WANTED_API_EXTS: HashSet<&'static CStr> = {
        [
            vkx::khr::Surface::name(),
            vkx::khr::Win32Surface::name(),
        ].as_ref().into_iter().map(|x| *x).collect()
    };
    
    static ref WANTED_DEV_EXTS: HashSet<&'static CStr> = {
        [
        ].as_ref().into_iter().map(|x| *x).collect()
    };
}

#[cfg(windows)]
#[derive(Clone)]
struct SurfaceConfig {
    hinst: *mut std::ffi::c_void,
    hwnd: *mut std::ffi::c_void,
}
impl SurfaceConfig {
    #[cfg(windows)]
    pub fn new(wnd: &Window) -> SurfaceConfig {
        use winit::platform::windows::WindowExtWindows;
        SurfaceConfig {
            hinst: wnd.hinstance(),
            hwnd: wnd.hwnd(),
        }
    }
}

#[derive(Default, Clone)]
pub struct InterfaceConfig {
    name: &'static str,
    flags: vk::QueueFlags,
    surf_cfg: Option<SurfaceConfig>,
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
    pub fn require_present(mut self, wnd: &Window) -> Self {
        self.surf_cfg = Some(SurfaceConfig::new(wnd));
        self
    }
    pub fn require_compute(mut self) -> Self {
        self.flags |= vk::QueueFlags::COMPUTE;
        self
    }
}

fn filter_exts(ext_props: &[vk::ExtensionProperties], wanted_exts: &HashSet<&'static CStr>) -> HashSet<&'static CStr> {
    ext_props.iter()
        .filter_map(|ext_prop| {
            if ext_prop.spec_version <= API_VERSION {
                let name = unsafe { CStr::from_ptr(ext_prop.extension_name.as_ptr()) };
                wanted_exts.get(name).map(|x| *x)
            } else { None }
        })
        .collect::<HashSet<_>>()
}

#[derive(Default)]
pub(crate) struct ApiExtensions {
    pub khr_surface: Option<vkx::khr::Surface>,
    #[cfg(windows)]
    pub khr_win32_surface: Option<vkx::khr::Win32Surface>,
}
impl ApiExtensions {
    fn new(entry: &ash::Entry, inst: &ash::Instance, enabled_exts: &HashSet<&CStr>) -> ApiExtensions {
        let mut rv = ApiExtensions::default();
        if enabled_exts.contains(vkx::khr::Surface::name()) {
            rv.khr_surface = Some(vkx::khr::Surface::new(entry, inst));
        }
        if enabled_exts.contains(vkx::khr::Win32Surface::name()) {
            rv.khr_win32_surface = Some(vkx::khr::Win32Surface::new(entry, inst));
        }
        rv
    }
}

pub(crate) struct ContextInner {
    // Don't change the order. Things should be dropped from top to bottom.
    pub inst: ash::Instance,
    pub entry: ash::Entry,
    pub api_exts: ApiExtensions,
}
pub struct Context(Arc<ContextInner>);
impl Context {
    pub fn new(app_name: &'static str) -> Result<Context> {
        info!("creating context for application '{}'", app_name);
        let entry = ash::Entry::new()?;
        let app_name = CString::new(app_name).unwrap();
        let engine_name = CString::new("insdraw").unwrap();
        // Extensions.
        let ext_props = unsafe { entry.enumerate_instance_extension_properties() }?;
        let api_exts = filter_exts(&ext_props, &WANTED_API_EXTS);
        trace!("wanted and supported instance extensions: {:?}", api_exts);

        let app_info = vk::ApplicationInfo::builder()
            .api_version(API_VERSION)
            .application_name(&app_name)
            .application_version(vk_make_version!(0, 0, 1))
            .engine_name(&engine_name)
            .engine_version(vk_make_version!(0, 0, 1))
            .build();
        let api_exts_raw = api_exts.iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();
        let inst_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&api_exts_raw)
            .build();
        let inst = unsafe { entry.create_instance(&inst_create_info, None)? };
        let api_exts = ApiExtensions::new(&entry, &inst, &api_exts);
        info!("created vulkan instance");

        let inner = ContextInner { entry, inst, api_exts };
        Ok(Context(Arc::new(inner)))
    }
    pub fn physdevs(&self) -> impl Iterator<Item=PhysicalDevice> + '_ {
        let ctxt = self.0.clone();
        if let Ok(physdevs) = unsafe { self.0.inst.enumerate_physical_devices() } {
            info!("discovered {} physical devices", physdevs.len());
            physdevs
        } else {
            warn!("unable to enumerate physical devices");
            Vec::default()
        }.into_iter()
            .map(move |physdev| PhysicalDevice::new(ctxt.clone(), physdev))
    }
}


pub struct PhysicalDevice {
    pub(crate) ctxt: Arc<ContextInner>,
    pub handle: vk::PhysicalDevice,
    pub prop: vk::PhysicalDeviceProperties,
    pub qfam_props: Vec<vk::QueueFamilyProperties>,
}
impl PhysicalDevice {
    fn new(ctxt: Arc<ContextInner>, handle: vk::PhysicalDevice) -> PhysicalDevice {
        // Assign physical device queues.
        let prop = unsafe { ctxt.inst.get_physical_device_properties(handle) };
        let qfam_props = unsafe { ctxt.inst.get_physical_device_queue_family_properties(handle) };
        PhysicalDevice { ctxt, handle, prop, qfam_props }
    }
}

#[derive(Default)]
pub struct DeviceExtensions {
}
impl DeviceExtensions {
    pub fn new(inst: &ash::Instance, dev: &ash::Device, enabled_exts: &HashSet<&'static CStr>) -> DeviceExtensions {
        let mut rv = DeviceExtensions::default();
        rv
    }
}

pub(crate) struct DeviceInner {
    pub(crate) ctxt: Arc<ContextInner>,
    pub dev: ash::Device,
    pub queues: HashMap<&'static str, vk::Queue>,
    pub dev_exts: HashSet<&'static CStr>,
    pub feats: vk::PhysicalDeviceFeatures,
}
pub struct Device(Arc<DeviceInner>);
impl Device {
    pub fn new(physdev: &PhysicalDevice, icfgs: &[InterfaceConfig]) -> Result<Device> {
        let ctxt = physdev.ctxt.clone();
        let dev_name = unsafe { CStr::from_ptr(physdev.prop.device_name.as_ptr()).to_string_lossy() };
        info!("checking '{}' ({:?})...", dev_name, physdev.prop.device_type);
        // Sort available queues by flag complexity in descending order. Complex
        // requirements should be met first.
        let mut icfgs = icfgs.to_vec();
        icfgs.sort_by_key(|x| Reverse(x.flags.as_raw().count_ones()));
        let (queue_create_infos, interface_qfam_idxs) = Self::derive_queues(physdev, &icfgs)?;

        // Features.
        let feats = vk::PhysicalDeviceFeatures::default();
        // Extensions.
        let ext_props = unsafe { ctxt.inst.enumerate_device_extension_properties(physdev.handle) }?;
        let dev_exts = filter_exts(&ext_props, &WANTED_DEV_EXTS);
        trace!("wanted and supported device extensions: {:?}", dev_exts);

        let dev_exts_raw = dev_exts.iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();
        let dev_create_info = vk::DeviceCreateInfo::builder()
            .enabled_features(&feats)
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&dev_exts_raw)
            .build();
        let dev = unsafe { ctxt.inst.create_device(physdev.handle, &dev_create_info, None)? };
        // Extract queues.
        let queues = interface_qfam_idxs.into_iter()
            .map(|(i, j, k)| {
                let name = icfgs[k].name;
                info!("set up queue for interface '{}'", name);
                let queue = unsafe { dev.get_device_queue(i as u32, j) };
                (name, queue)
            })
            .collect::<HashMap<_, _>>();
        info!("created device on '{}' ({:?})", dev_name, physdev.prop.device_type);

        let inner = DeviceInner { ctxt, dev, queues, feats, dev_exts };
        Ok(Device(Arc::new(inner)))
    }
    fn derive_queues(physdev: &PhysicalDevice, icfgs: &[InterfaceConfig]) -> Result<(Vec<vk::DeviceQueueCreateInfo>, Vec<(usize, u32, usize)>)> {
        // In following codes, `i` is queue family index; `j` is queue index in
        // a family; and `k` is the index of interface.

        // Assign physical device queues.
        let mut qfam_caps = physdev.qfam_props.iter()
            .map(|x| x.queue_flags)
            .enumerate()
            .collect::<Vec<_>>();
        // Sort available queues by flag complexity in ascending order.
        // Versatile queues should be kept for more complicated requirements.
        qfam_caps.sort_by_key(|&(_i, flags)| flags.as_raw().count_ones());
        let qfam_counts = physdev.qfam_props.iter()
            .map(|x| x.queue_count)
            .collect::<Vec<_>>();
        let mut qfam_queue_idxs = repeat(0)
            .take(qfam_counts.len())
            .collect::<Vec<_>>();
        let interface_qfam_idxs = icfgs.iter()
            .enumerate()
            .filter_map(|(k, icfg)| {
                // Find a simplest queue that meets the requirement of current
                // interface config.
                qfam_caps.iter()
                    .find_map(|&(i, flags)| {
                        let j = qfam_queue_idxs[i];
                        // Test whether the queue family support presenting, if
                        // present is required.
                        if let Some(surf_cfg) = &icfg.surf_cfg {
                            // physdev.ctxt.api_exts.khr_surface. // TODO:
                        }
                        if flags.contains(icfg.flags) && qfam_counts[i] > j {
                            qfam_queue_idxs[i] += 1;
                            Some((i, j, k))
                        } else { None }
                    })
            })
            .collect::<Vec<_>>();
        if interface_qfam_idxs.len() != icfgs.len() {
            return Err(Error::NoCapablePhysicalDevice);
        }
        // Mapping from queue family index to queue priorities.
        let mut priorities = HashMap::<usize, Vec<f32>>::new();
        for &(i, _, k) in interface_qfam_idxs.iter() {
            priorities.entry(i).or_default().push(icfgs[k].priority);
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
        Ok((create_infos, interface_qfam_idxs))
    }
}


pub struct ShaderModuleInner {
    pub(crate) dev: Arc<DeviceInner>,
    pub handle: vk::ShaderModule,
    pub entry_points: HashMap<String, EntryPointManifest>,
}
pub struct ShaderModule(Arc<ShaderModuleInner>);
impl ShaderModule {
    pub fn new(dev: &Device, spv: &SpirvBinary) -> Result<ShaderModule> {
        let dev = dev.0.clone();
        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(spv.words())
            .build();
        let handle = unsafe { dev.dev.create_shader_module(&create_info, None)? };
        let entry_points = spv.reflect()?
            .into_iter()
            .map(|x| (x.name.to_owned(), x.clone()))
            .collect::<HashMap<_, _>>();
        let inner = ShaderModuleInner { dev, handle, entry_points };
        info!("created shader module");
        Ok(ShaderModule(Arc::new(inner)))
    }
    pub fn entry_points(&self) -> impl Iterator<Item=&EntryPointManifest> {
        self.0.entry_points.values()
    }
    pub fn get(&self, entry_point_name: &str) -> Option<&EntryPointManifest> {
        self.0.entry_points.get(entry_point_name)
    }
}
impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe { self.0.dev.dev.destroy_shader_module(self.0.handle, None) };
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
