use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, Weak};
use std::ops::Range;
use log::{info, warn, error, debug, trace};
use ash::vk;
use ash::vk_make_version;
use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use ash::extensions as vkx;
use lazy_static::lazy_static;
use spirq::{SpirvBinary, EntryPoint as EntryPointManifest, InterfaceLocation,
    Manifest};
use spirq::ty::{DescriptorType, Type};
use winit::window::Window;
use super::error::{Error, Result};
use super::alloc::BuddyAllocator;

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
            vkx::khr::Swapchain::name(),
        ].as_ref().into_iter().map(|x| *x).collect()
    };
}

fn filter_exts(
    ext_props: &[vk::ExtensionProperties],
    wanted_exts: &HashSet<&'static CStr>
) -> HashSet<&'static CStr> {
    ext_props.iter()
        .filter_map(|ext_prop| {
            if ext_prop.spec_version <= API_VERSION {
                let name = unsafe {
                    CStr::from_ptr(ext_prop.extension_name.as_ptr())
                };
                wanted_exts.get(name).map(|x| *x)
            } else { None }
        })
        .collect::<HashSet<_>>()
}

macro_rules! try_ext {
    ($a: expr, $b: expr, $enabled_exts: expr, $ext_ty: ty) => {
        if $enabled_exts.contains(<$ext_ty>::name()) {
            Ok(<$ext_ty>::new($a, $b))
        } else { Err(Error::MissingExtension(<$ext_ty>::name())) }
    }
}

macro_rules! def_exts {
    ($ty:ident ($a_name:ident: $a_ty:ty, $b_name:ident: $b_ty:ty) => {$($ext_name:ident: $ext_ty:ty,)*}) => {
        pub(crate) struct $ty {
            $( $ext_name: Option<$ext_ty>, )*
        }
        impl $ty {
            fn new($a_name: &$a_ty, $b_name: &$b_ty, enabled_exts: &HashSet<&CStr>) -> $ty {
                let mut rv = unsafe { MaybeUninit::<Self>::uninit().assume_init() };
                $(
                    if enabled_exts.contains(<$ext_ty>::name()) {
                        rv.$ext_name = Some(<$ext_ty>::new($a_name, $b_name))
                    };
                )*
                rv
            }
            $(
                fn $ext_name(&self) -> Result<&$ext_ty> {
                    self.$ext_name.as_ref()
                        .ok_or(Error::MissingExtension(<$ext_ty>::name()))
                }
            )*
        }
    }
}

def_exts! {
    ApiExtensions(entry: ash::Entry, inst: ash::Instance) => {
        khr_surface: vkx::khr::Surface,
        khr_win32_surface: vkx::khr::Win32Surface,
    }
}

def_exts! {
    DeviceExtensions(inst: ash::Instance, dev: ash::Device) => {
        khr_swapchain: vkx::khr::Swapchain,
    }
}

macro_rules! impl_ptr_wrapper {
    ($ptr_ty: ident -> $inner_ty: ident) => {
        impl std::ops::Deref for $ptr_ty {
            type Target = $inner_ty;
            fn deref(&self) -> &Self::Target { &*self.0 }
        }
        impl std::clone::Clone for $ptr_ty {
            fn clone(&self) -> $ptr_ty { $ptr_ty(self.0.clone()) }
        }
    }
}


fn fmt2aspect(fmt: vk::Format) -> Result<vk::ImageAspectFlags> {
    // TODO: Check this format code limit.
    if fmt.as_raw() >= 1000156000 {
        return Err(Error::UnsupportedPlatform);
    }
    let aspect = match fmt {
        vk::Format::UNDEFINED => Default::default(),
        vk::Format::D16_UNORM |
            vk::Format::X8_D24_UNORM_PACK32 |
            vk::Format::D32_SFLOAT => { vk::ImageAspectFlags::DEPTH },
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT |
            vk::Format::D24_UNORM_S8_UINT |
            vk::Format::D32_SFLOAT_S8_UINT =>
        { vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL },
        _ => vk::ImageAspectFlags::COLOR,
    };
    Ok(aspect)
}


pub struct ContextInner {
    // Don't change the order. Things should be dropped from top to bottom.
    inst: ash::Instance,
    entry: ash::Entry,
    cap_detail: ContextCapabilityDetail,
    surf_detail: Option<ContextSurfaceDetail>,
    physdevs: Vec<vk::PhysicalDevice>,
}
impl_ptr_wrapper!(Context -> ContextInner);
pub struct Context(Arc<ContextInner>);
impl Context {
    fn filter_api_exts(entry: &ash::Entry) -> Result<HashSet<&'static CStr>> {
        let ext_props = unsafe {
            entry.enumerate_instance_extension_properties()
        }?;
        let api_exts = filter_exts(&ext_props, &WANTED_API_EXTS);
        trace!("wanted and supported instance extensions: {:?}", api_exts);
        Ok(api_exts)
    }
    fn create_inst(
        entry: &ash::Entry,
        app_name: &str,
        api_exts: &HashSet<&'static CStr>,
    ) -> Result<ash::Instance> {
        let app_name = CString::new(app_name).unwrap();
        let engine_name = CString::new("insdraw").unwrap();
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
        info!("created vulkan instance");
        Ok(inst)
    }
    pub fn new(app_name: &str, wnd: Option<&Window>) -> Result<Context> {
        let entry = ash::Entry::new()?;
        let api_exts = Self::filter_api_exts(&entry)?;
        let inst = Self::create_inst(&entry, app_name, &api_exts)?;
        info!("created context for application '{}'", app_name);

        let cap_detail = ContextCapabilityDetail::new(&entry, &inst, &api_exts);
        let surf_detail = if let Some(wnd) = wnd {
            let surf_detail = ContextSurfaceDetail::new(&cap_detail, wnd)?;
            Some(surf_detail)
        } else { None };
        let physdevs = unsafe { inst.enumerate_physical_devices()? };
        info!("discovered {} physical devices", physdevs.len());

        let inner = ContextInner {
            entry, inst, cap_detail, surf_detail, physdevs
        };
        Ok(Context(Arc::new(inner)))
    }

    pub fn physdevs(&self) -> impl Iterator<Item=Result<PhysicalDevice>> + '_ {
        (0..self.physdevs.len()).into_iter()
            .map(move |i| {
                match PhysicalDevice::new(self, i) {
                    Ok(physdev) => {
                        info!("device #{}: {:?}", i, physdev);
                        Ok(physdev)
                    },
                    Err(e) => {
                        info!("device #{} not accessible", i);
                        Err(e)
                    },
                }
            })
    }
}
impl Drop for ContextInner {
    fn drop(&mut self) {
        if let Some(surf_detail) = self.surf_detail.as_ref() {
            surf_detail.wipe(&self.cap_detail);
        }
        unsafe { self.inst.destroy_instance(None) };
    }
}

struct ContextCapabilityDetail {
    api_exts: ApiExtensions,
}
impl ContextCapabilityDetail {
    pub fn new(
        entry: &ash::Entry,
        inst: &ash::Instance,
        api_exts: &HashSet<&'static CStr>,
    ) -> ContextCapabilityDetail {
        let api_exts = ApiExtensions::new(&entry, &inst, &api_exts);
        ContextCapabilityDetail { api_exts }
    }
}

struct ContextSurfaceDetail {
    surf: vk::SurfaceKHR,
    width: u32,
    height: u32,
}
impl ContextSurfaceDetail {
    fn create_surf(
        ctxt_cap_detail: &ContextCapabilityDetail,
        wnd: &Window,
    ) -> Result<vk::SurfaceKHR> {
        if cfg!(windows) {
            use winit::platform::windows::WindowExtWindows;
            let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                .hinstance(wnd.hinstance())
                .hwnd(wnd.hwnd())
                .build();
            let surf = unsafe {
                ctxt_cap_detail.api_exts.khr_win32_surface()?
                    .create_win32_surface(&create_info, None)?
            };
            Ok(surf)
        } else { Err(Error::UnsupportedPlatform) }
    }
    pub fn new(
        ctxt_cap_detail: &ContextCapabilityDetail,
        wnd: &Window,
    ) -> Result<ContextSurfaceDetail> {
        let surf = Self::create_surf(&*ctxt_cap_detail, wnd)?;
        let (width, height): (u32, u32) = wnd.inner_size()
            .to_physical(wnd.hidpi_factor())
            .into();
        let surf_detail = ContextSurfaceDetail { surf, width, height };
        Ok(surf_detail)
    }
    pub fn wipe(&self, ctxt_cap_detail: &ContextCapabilityDetail) {
        unsafe {
            ctxt_cap_detail.api_exts.khr_surface().unwrap()
                .destroy_surface(self.surf, None);
        }
    }
}

pub struct QueueFamily {
    qfam_idx: u32,
    nqueue: u32,
    qflags: vk::QueueFlags,
    present: bool,
}
impl QueueFamily {
    pub fn new(
        ctxt: &Context,
        physdev: vk::PhysicalDevice,
        qfam_idx: u32,
        qfam_prop: &vk::QueueFamilyProperties,
    ) -> QueueFamily {
        let nqueue = qfam_prop.queue_count;
        let qflags = qfam_prop.queue_flags;
        let present = ctxt.cap_detail.api_exts.khr_surface.as_ref()
            .and_then(|khr_surface| {
                let surf_detail = ctxt.surf_detail.as_ref()?;
                let support = unsafe {
                    khr_surface.get_physical_device_surface_support(physdev,
                        qfam_idx, surf_detail.surf)
                };
                Some(support)
            }).unwrap_or(false);
        QueueFamily { qfam_idx, nqueue, qflags, present }
    }
}

pub struct PhysicalDeviceInner {
    ctxt: Context,
    physdev: vk::PhysicalDevice,
    pub prop: vk::PhysicalDeviceProperties,
    pub mem_prop: vk::PhysicalDeviceMemoryProperties,
    pub qfams: Vec<QueueFamily>,
    pub dev_exts: HashSet<&'static CStr>,
    pub feats: vk::PhysicalDeviceFeatures,
    surf_detail: Option<PhysicalDeviceSurfaceDetail>,
}
impl_ptr_wrapper!(PhysicalDevice -> PhysicalDeviceInner);
pub struct PhysicalDevice(Arc<PhysicalDeviceInner>);
impl PhysicalDevice {
    fn filter_dev_exts(
        ctxt: &Context,
        physdev: vk::PhysicalDevice,
    ) -> Result<HashSet<&'static CStr>> {
        let ext_props = unsafe {
            ctxt.inst.enumerate_device_extension_properties(physdev)?
        };
        let dev_exts = filter_exts(&ext_props, &WANTED_DEV_EXTS);
        trace!("wanted and supported device extensions: {:?}", dev_exts);
        Ok(dev_exts)
    }
    fn filter_feats(
        ctxt: &Context,
        physdev: vk::PhysicalDevice,
    ) -> Result<vk::PhysicalDeviceFeatures> {
        let feats = vk::PhysicalDeviceFeatures::default();
        Ok(feats)
    }
    pub fn new(ctxt: &Context, iphysdev: usize) -> Result<PhysicalDevice> {
        let physdev = *ctxt.physdevs.get(iphysdev)
            .ok_or(Error::InvalidOperation)?;

        let dev_exts = Self::filter_dev_exts(ctxt, physdev)?;
        let feats = Self::filter_feats(ctxt, physdev)?;
    
        let prop = unsafe { ctxt.inst.get_physical_device_properties(physdev) };
        let mem_prop = unsafe {
            ctxt.inst.get_physical_device_memory_properties(physdev)
        };
        let qfam_props = unsafe {
            ctxt.inst.get_physical_device_queue_family_properties(physdev)
        };
        let qfams = qfam_props.into_iter()
            .enumerate()
            .map(|(i, prop)| QueueFamily::new(ctxt, physdev, i as u32, &prop))
            .collect();
        let surf_detail = if ctxt.surf_detail.is_some() {
            let surf_detail = PhysicalDeviceSurfaceDetail::new(ctxt, physdev)?;
            Some(surf_detail)
        } else { None };
        let ctxt = ctxt.clone();
        let inner = PhysicalDeviceInner {
            ctxt, physdev, prop, mem_prop, qfams, dev_exts, feats, surf_detail
        };
        Ok(PhysicalDevice(Arc::new(inner)))
    }
}
impl std::fmt::Debug for PhysicalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let dev_name = unsafe {
            CStr::from_ptr(self.prop.device_name.as_ptr()).to_string_lossy()
        };
        let dev_ty = self.prop.device_type;
        write!(f, "'{}' ({:?})", dev_name, dev_ty)
    }
}

struct PhysicalDeviceSurfaceDetail {
    surf_cap: vk::SurfaceCapabilitiesKHR,
    surf_fmt: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
}
impl PhysicalDeviceSurfaceDetail {
    fn sel_surf_fmt(
        khr_surface: &vkx::khr::Surface,
        physdev: vk::PhysicalDevice,
        surf: vk::SurfaceKHR,
    ) -> Result<vk::SurfaceFormatKHR> {
        let surf_fmts = unsafe {
            khr_surface.get_physical_device_surface_formats(physdev, surf)?
        };
        let fallback = surf_fmts[0].clone();
        let surf_fmt = surf_fmts.into_iter()
            .find(|surf_fmt| surf_fmt.format == vk::Format::B8G8R8A8_UNORM)
            .unwrap_or(fallback);
        Ok(surf_fmt)
    }
    fn sel_present_mode(
        khr_surface: &vkx::khr::Surface,
        physdev: vk::PhysicalDevice,
        surf: vk::SurfaceKHR,
    ) -> Result<vk::PresentModeKHR> {
        let present_modes = unsafe {
            khr_surface.get_physical_device_surface_present_modes(physdev,
                surf)?
        };
        let present_mode = present_modes.into_iter()
            .find(|&present_mode| present_mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);
        Ok(present_mode)
    }
    pub fn new(
        ctxt: &Context,
        physdev: vk::PhysicalDevice,
    ) -> Result<PhysicalDeviceSurfaceDetail> {
        let khr_surface = ctxt.cap_detail.api_exts.khr_surface.as_ref()
            .ok_or(Error::InvalidOperation)?;
        let surf = ctxt.surf_detail.as_ref()
            .ok_or(Error::InvalidOperation)?
            .surf;
        let surf_cap = unsafe {
            khr_surface.get_physical_device_surface_capabilities(physdev, surf)?
        };
        let surf_fmt = Self::sel_surf_fmt(khr_surface, physdev, surf)?;
        let present_mode = Self::sel_present_mode(khr_surface, physdev, surf)?;
        let physdev_surf_detail = PhysicalDeviceSurfaceDetail {
            surf_cap, surf_fmt, present_mode };
        Ok(physdev_surf_detail)
    }
}


pub struct DeviceInner {
    physdev: PhysicalDevice,
    dev: ash::Device,
    cap_detail: DeviceCapabilityDetail,
    malloc_detail: DeviceMemoryAllocationDetail,
    present_detail: Option<DevicePresentationDetail>,
}
impl_ptr_wrapper!(Device -> DeviceInner);
pub struct Device(Arc<DeviceInner>);
impl Device {
    fn create_dev(
        physdev: &PhysicalDevice,
        dev_exts: &HashSet<&'static CStr>,
        feats: &vk::PhysicalDeviceFeatures,
        qalloc: &QueueAllocation,
    ) -> Result<ash::Device> {
        let dqcis = qalloc.qfam_alloced.iter()
            .map(|(&qfam_idx, &nqueue)| {
                let priors = (0..nqueue).into_iter()
                    .map(|_| 0.5)
                    .collect::<Vec<_>>();
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(qfam_idx)
                    .queue_priorities(&priors)
                    .build()
            })
            .collect::<Vec<_>>();
        let dev_exts_raw = dev_exts.iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();
        let create_info = vk::DeviceCreateInfo::builder()
            .enabled_features(&feats)
            .queue_create_infos(&dqcis)
            .enabled_extension_names(&dev_exts_raw)
            .build();
        let dev = unsafe {
            physdev.ctxt.inst.create_device(physdev.physdev, &create_info,
                None)?
        };
        info!("created device on {:?}", physdev);
        Ok(dev)
    }
    fn collect_qmap(
        dev: &ash::Device,
        qalloc: &QueueAllocation,
    ) -> HashMap<QueueInterface, vk::Queue> {
        qalloc.qloc_assign.iter()
            .filter_map(|(qi, qloc)| {
                if let Some(qloc) = qloc {
                    info!("set up queue for interface '{:?}'", qi);
                    let queue = unsafe {
                        dev.get_device_queue(qloc.qfam_idx, qloc.queue_idx)
                    };
                    Some((*qi, queue))
                } else {
                    warn!("interface '{:?}' is not available", qi);
                    None
                }
            })
            .collect::<HashMap<_, _>>()
    }
    pub fn new(physdev: &PhysicalDevice) -> Result<Device> {
        let qalloc = QueueAllocation::new(physdev);
        let dev = Self::create_dev(physdev, &physdev.dev_exts, &physdev.feats,
            &qalloc)?;

        let qmap = Self::collect_qmap(&dev, &qalloc);

        let cap_detail = {
            DeviceCapabilityDetail::new(physdev, &dev, &physdev.dev_exts,
                &physdev.feats)
        };
        let malloc_detail = {
            let trans_queue = qmap.get(&QueueInterface::Transfer)
                .map(|x| *x);
            DeviceMemoryAllocationDetail::new(physdev, &dev, trans_queue)?
        };
        let present_detail = {
            if let Some(&present_queue) = qmap.get(&QueueInterface::Present) {
                let present_detail = DevicePresentationDetail::new(&physdev,
                    &cap_detail, present_queue)?;
                Some(present_detail)
            } else { None }
        };
        let physdev = physdev.clone();
        let inner = DeviceInner { physdev, dev, cap_detail,
            malloc_detail, present_detail };
        Ok(Device(Arc::new(inner)))
    }

    /// Allocate memory on the device. Small memory chunks will be paged while
    /// large memory chunks are provided with dedicated allocation.
    fn alloc_mem(
        dev: &Device,
        mem_req: &vk::MemoryRequirements,
        mem_usage: MemoryUsage,
    ) -> Result<MemorySlice> {
        let size = mem_req.size as usize;
        let align = mem_req.alignment as usize;
        let mem_ty = &dev.malloc_detail.mem_ty_sel[&mem_usage]
            .match_mem_ty(mem_req.memory_type_bits)
            .ok_or(Error::UnsupportedPlatform)?;
        if size > PagedMemoryAllocator::PAGE_SIZE {
            // Dedicated allocation.
            let mem = Arc::new(Memory::new(dev, mem_ty, size, None)?);
            Ok(MemorySlice::new(&mem, 0, size))
        } else {
            // Paged allocation.
            let weak_dev = Arc::downgrade(&dev.0);
            dev.malloc_detail.pallocs
                .lock()
                .unwrap()
                .entry(mem_ty.mem_ty_idx)
                .or_insert_with(|| PagedMemoryAllocator::new(mem_ty))
                .alloc(weak_dev, size, align)
        }
    }
    fn alloc_buf_mem(
        dev: &Device,
        buf: vk::Buffer,
        mem_usage: MemoryUsage,
    ) -> Result<MemorySlice> {
        let mem_req = unsafe { dev.dev.get_buffer_memory_requirements(buf) };
        Self::alloc_mem(dev, &mem_req, mem_usage)
            .and_then(|mem| unsafe {
                let offset = mem.offset as u64;
                let dev_mem = mem.mem.dev_mem;
                dev.dev.bind_buffer_memory(buf, dev_mem, offset)?;
                Ok(mem)
            })
    }
    fn alloc_img_mem(
        dev: &Device,
        img: vk::Image,
        mem_usage: MemoryUsage,
    ) -> Result<MemorySlice> {
        let mem_req = unsafe { dev.dev.get_image_memory_requirements(img) };
        Self::alloc_mem(dev, &mem_req, mem_usage)
            .and_then(|mem| unsafe {
                let offset = mem.offset as u64;
                let dev_mem = mem.mem.dev_mem;
                dev.dev.bind_image_memory(img, dev_mem, offset)?;
                Ok(mem)
            })
    }
}
#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
enum QueueInterface {
    Graphics,
    Compute,
    Transfer,
    Present,
}
struct QueueLocation {
    /// Index of the queue family of the queue.
    qfam_idx: u32,
    /// Index of the queue within the queue family.
    queue_idx: u32,
}
struct QueueAllocation {
    /// Location assigned to queue interfaces.
    qloc_assign: HashMap<QueueInterface, Option<QueueLocation>>,
    /// Mapping from queue family index to number of queues allocated in the
    /// family.
    qfam_alloced: HashMap<u32, u32>,
}
impl QueueAllocation {
    fn collect_qfam_idxs_by_cap(
        physdev: &PhysicalDevice,
    ) -> [(QueueInterface, HashSet<u32>); 4] {
        let qfams = &physdev.qfams;
        let nqfam = qfams.len();
        let graph_qfam_idxs = (0..nqfam).into_iter()
            .filter_map(|i| {
                if qfams[i].qflags.intersects(vk::QueueFlags::GRAPHICS) {
                    Some(i as u32)
                } else { None }
            })
            .collect::<HashSet<_>>();
        let comp_qfam_idxs = (0..nqfam).into_iter()
            .filter_map(|i| {
                if qfams[i].qflags.intersects(vk::QueueFlags::COMPUTE) {
                    Some(i as u32)
                } else { None }
            })
            .collect::<HashSet<_>>();
        let trans_qfam_idxs = (0..nqfam).into_iter()
            .filter_map(|i| {
                if qfams[i].qflags.intersects(vk::QueueFlags::TRANSFER) {
                    Some(i as u32)
                } else { None }
            })
            .collect::<HashSet<_>>();
        let present_qfam_idxs = (0..nqfam).into_iter()
            .filter_map(|i| {
                if qfams[i].present { Some(i as u32) } else { None }
            })
            .collect::<HashSet<_>>();
        [
            (QueueInterface::Graphics, graph_qfam_idxs),
            (QueueInterface::Compute, comp_qfam_idxs),
            (QueueInterface::Transfer, trans_qfam_idxs),
            (QueueInterface::Present, present_qfam_idxs),
        ]
    }
    pub fn new(physdev: &PhysicalDevice) -> QueueAllocation {
        // Mapping from queue family to the number of queues has been allocated
        // in that family.
        let mut qloc_assign: HashMap<QueueInterface, Option<QueueLocation>> = {
            HashMap::new()
        };
        let mut qfam_alloced: HashMap<u32, u32> = HashMap::new();
        let mut cap_qfams = Self::collect_qfam_idxs_by_cap(physdev);
        // Satify scarce resource requirements first.
        cap_qfams.sort_by_key(|x| x.1.len());
        for cap_qfam in cap_qfams.iter() {
            let qi = cap_qfam.0;
            // Find a queue family that hasn't been filled up.
            if let Some(qloc) = cap_qfam.1.iter()
                .find_map(|&qfam_idx| {
                    let nqueue = physdev.qfams[qfam_idx as usize].nqueue;
                    let queue_idx = *qfam_alloced.get(&qfam_idx).unwrap_or(&0);
                    if queue_idx < nqueue {
                        Some(QueueLocation { qfam_idx, queue_idx })
                    } else { None }
                })
            {
                // Allocate available queue to the capability.
                *qfam_alloced.entry(qloc.qfam_idx).or_default() += 1;
                qloc_assign.insert(qi, Some(qloc));
            } else {
                qloc_assign.insert(qi, None);
            }
        }
        QueueAllocation { qloc_assign, qfam_alloced }
    }
}

struct DeviceCapabilityDetail {
    /// Enabled wanted device extensions.
    dev_exts: DeviceExtensions,
    /// Enabled wanted device features.
    feats: vk::PhysicalDeviceFeatures,
}
impl DeviceCapabilityDetail {
    pub fn new(
        physdev: &PhysicalDevice,
        dev: &ash::Device,
        dev_exts: &HashSet<&'static CStr>,
        feats: &vk::PhysicalDeviceFeatures,
    ) -> DeviceCapabilityDetail {
        let dev_exts = DeviceExtensions::new(&physdev.ctxt.inst, dev, dev_exts);
        let feats = feats.clone();
        DeviceCapabilityDetail { dev_exts, feats }
    }
}

// ------ Memory allocation ------

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct MemoryProperty(vk::MemoryPropertyFlags);
impl MemoryProperty {
    #[inline]
    fn device_local_bit(&self) -> u32 {
        (self.0.as_raw() / vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw()) & 1
    }
    #[inline]
    fn host_cached_bit(&self) -> u32 {
        (self.0.as_raw() / vk::MemoryPropertyFlags::HOST_CACHED.as_raw()) & 1
    }
    #[inline]
    fn host_visible_bit(&self) -> u32 {
        (self.0.as_raw() / vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()) & 1
    }
    #[inline]
    fn host_coherent_bit(&self) -> u32 {
        (self.0.as_raw() / vk::MemoryPropertyFlags::HOST_COHERENT.as_raw()) & 1
    }

    pub fn push_score(&self) -> u32 {
        self.host_visible_bit() << 3 |
            self.host_coherent_bit() << 2 |
            self.device_local_bit() << 1 |
            self.host_cached_bit()
    }
    pub fn pull_score(&self) -> u32 {
        self.host_visible_bit() << 3 |
            self.host_cached_bit() << 2 |
            self.host_coherent_bit() << 1 |
            self.device_local_bit()
    }
    pub fn dev_score(&self) -> u32 {
        self.device_local_bit() << 3 |
            self.host_visible_bit() << 2 |
            self.host_coherent_bit() << 1 |
            self.host_cached_bit()
    }
}
#[derive(Debug, Clone)]
struct MemoryType {
    mem_ty_idx: u32,
    mem_prop: MemoryProperty,
}


struct MemoryInner {
    dev: Device,
    dev_mem: vk::DeviceMemory,
    size: usize,
    mem_prop: MemoryProperty,
    malloc: Option<Mutex<BuddyAllocator>>,
}
impl_ptr_wrapper!(Memory -> MemoryInner);
struct Memory(Arc<MemoryInner>);
impl Memory {
    fn alloc_dev_mem(
        dev: &Device,
        mem_ty: &MemoryType,
        size: usize,
    ) -> Result<vk::DeviceMemory> {
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size as u64)
            .memory_type_index(mem_ty.mem_ty_idx)
            .build();
        info!("allocated device memory of {} bytes", size);
        let dev_mem = unsafe { dev.dev.allocate_memory(&alloc_info, None)? };
        Ok(dev_mem)
    }
    fn new(
        dev: &Device,
        mem_ty: &MemoryType,
        size: usize,
        malloc: Option<BuddyAllocator>,
    ) -> Result<Memory> {
        let dev = dev.clone();
        let dev_mem = Self::alloc_dev_mem(&dev, mem_ty, size)?;
        let mem_prop = mem_ty.mem_prop.clone();
        let malloc = malloc.map(|x| Mutex::new(x));
        let inner = MemoryInner { dev, dev_mem, size, mem_prop, malloc };
        Ok(Memory(Arc::new(inner)))
    }
}
impl Drop for MemoryInner {
    fn drop(&mut self) {
        unsafe { self.dev.dev.free_memory(self.dev_mem, None) };
    }
}

/// A section of memory in a large allocation.
#[derive(Clone)]
pub struct MemorySlice {
    mem: Memory,
    /// Offset of the suballocation in the owner memory page. `page_offset` is
    /// always less than any current offset of the slice; and `offset + size` is
    /// always within the bound of the allocated range. E.g.
    ///
    /// ```
    /// page_offset->|<- allocated size ->|
    ///           offset->|<- size ->|
    /// ```
    page_offset: usize,
    offset: usize,
    size: usize,
}
impl MemorySlice {
    /// Create a new memory slice and refer to the `offset` if `mem` is paged.
    /// This should be used internally only.
    fn new(mem: &Memory, offset: usize, size: usize) -> MemorySlice {
        if let Some(malloc) = mem.malloc.as_ref() {
            // Increase the reference count at the referred intra-page location
            // immediately.
            malloc.lock().unwrap().refer(offset);
        }
        let mem = mem.clone();
        MemorySlice { mem, page_offset: offset, offset, size }
    }
    pub fn slice(&self, offset: usize, size: usize) -> Result<MemorySlice> {
        if offset + size <= self.size {
            let mem = MemorySlice {
                mem: self.mem.clone(),
                page_offset: self.page_offset,
                offset: self.offset + offset,
                size: size,
            };
            Ok(mem)
        } else {
            Err(Error::InvalidOperation)
        }
    }
    pub fn write(&self, src: &[u8]) -> Result<()> {
        let dev_mem = self.mem.dev_mem;
        let dev = &self.mem.dev.dev;
        unsafe {
            let offset = self.offset as u64;
            let dst = dev.map_memory(
                dev_mem,
                offset,
                self.size as u64,
                vk::MemoryMapFlags::empty())? as *mut u8;
            std::intrinsics::copy(src.as_ptr(), dst, self.size);
            dev.unmap_memory(dev_mem);
        }
        Ok(())
    }
    pub fn read(&self, dst: &mut [u8]) -> Result<()> {
        let dev_mem = self.mem.dev_mem;
        let dev = &self.mem.dev.dev;
        unsafe {
            let offset = self.offset as u64;
            let src = dev.map_memory(
                dev_mem,
                offset,
                self.size as u64,
                vk::MemoryMapFlags::empty())? as *mut u8;
            std::intrinsics::copy(src, dst.as_mut_ptr(), self.size);
            dev.unmap_memory(dev_mem);
        }
        Ok(())
    }
}
impl Drop for MemorySlice {
    fn drop(&mut self) {
        if let Some(malloc) = self.mem.malloc.as_ref() {
            // Decrease the reference count at the location.
            malloc.lock().unwrap().free(self.page_offset);
        }
    }
}

/// Manager of small object allocation.
struct PagedMemoryAllocator {
    pages: Vec<Memory>,
    pos: usize,
    mem_ty: MemoryType,
}
impl PagedMemoryAllocator {
    pub const PAGE_SIZE: usize = 8 * 1024 * 1024;
    pub const HALF_PAGE_SIZE: usize = Self::PAGE_SIZE / 2;
    pub const PAGE_MAX_ORDER: u32 = 15;

    pub fn new(mem_ty: &MemoryType) -> PagedMemoryAllocator {
        PagedMemoryAllocator {
            pages: Vec::new(),
            pos: 0,
            mem_ty: mem_ty.clone(),
        }
    }
    /// Allocate on memory page indexed by `i`.
    fn alloc_on(
        &mut self,
        i: usize,
        size: usize,
        align: usize,
    ) -> Option<MemorySlice> {
        let mut cur_page = &mut self.pages[i];
        let addr = {
            cur_page.malloc.as_ref().unwrap().lock().unwrap()
                .alloc(size, align)
        };
        addr.map(|offset| MemorySlice::new(&cur_page, offset, size))
    }
    /// Allocate on a newly created page, added at the end of the page list.
    /// This method doesn't take alignment because the suballocation will
    /// always taken at offset=0.
    fn alloc_on_new_page(
        &mut self,
        dev: Weak<DeviceInner>,
        size: usize,
    ) -> Result<MemorySlice> {
        // Out of paged memory. We have to allocate the memory on a new
        // page.
        let dev = dev.upgrade()
            .ok_or(Error::InvalidOperation)?;
        let mem = {
            let malloc = {
                Some(BuddyAllocator::new(Self::PAGE_SIZE, Self::PAGE_MAX_ORDER))
            };
            Memory::new(&Device(dev), &self.mem_ty, Self::PAGE_SIZE, malloc)?
        };
        self.pos = self.pages.len();
        self.pages.push(mem);
        info!("created memory page");
        // This should not fail, unless the allocation size is greater than
        // `PAGE_SIZE` which is not allowed.
        self.alloc_on(self.pos, size, 0)
            .ok_or(Error::InvalidOperation)
    }
    /// Allocate on memory page. A new page is allocated if the allocation
    /// cannot be done on existing pages. Size of memory to be allocated MUST be
    /// less than `PAGE_SIZE`.
    pub fn alloc(
        &mut self,
        dev: Weak<DeviceInner>,
        size: usize,
        align: usize,
    ) -> Result<MemorySlice> {
        // Attempt to allocatef on existing page.
        let front = self.pos..self.pages.len();
        let back = 0..self.pos;
        front.into_iter().chain(back.into_iter())
            .find_map(|i| self.alloc_on(i, size, align))
            .ok_or(Error::OutOfMemory)
            .or_else(|_| self.alloc_on_new_page(dev, size))
    }
    /// Drop all unused memory pages.
    pub fn shrink_to_fit(&mut self, dev: &Device) {
        let mut i = 0;
        while i < self.pages.len() {
            let unused = {
                let malloc = self.pages[i].malloc.as_ref().unwrap();
                let malloc = malloc.lock().unwrap();
                malloc.is_unused()
            };
            if unused {
                self.pages.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }
}

#[repr(u32)]
#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub enum MemoryUsage {
    /// Constant data container which is seldom written by the host but
    /// potentially frequently read by the device. I/O with device memory might
    /// incur staging memory allocations.
    ///
    /// NOTE: It's NOT guaranteed the memory will be allocated on a device-local
    /// heap.
    Device,
    /// Pushing data container which is frequently written by the host and
    /// potentially frequently read by the device.
    Push,
    /// Pulling data container which is frequently read by the host and
    /// potentially frequently written by the device.
    Pull,
}

struct MemoryTypeFallbackChain(Vec<MemoryType>);
impl MemoryTypeFallbackChain {
    fn match_mem_ty(&self, mem_ty_bits: u32) -> Option<&MemoryType> {
        self.0.iter()
            .find(|mem_ty| mem_ty_bits & 1 << mem_ty.mem_ty_idx != 0)
    }
}
#[derive(Default)]
struct MemoryStaging {
    trans_queue: vk::Queue,
    push: Mutex<Option<MemoryInner>>,
    pull: Mutex<Option<MemoryInner>>,
}
struct DeviceMemoryAllocationDetail {
    mem_ty_sel: HashMap<MemoryUsage, MemoryTypeFallbackChain>,
    stage: Option<MemoryStaging>,
    /// Allocator for each memory type. The indices of allocator correspond to
    /// memory type indices.
    pallocs: Mutex<HashMap<u32, PagedMemoryAllocator>>,
}
impl DeviceMemoryAllocationDetail {
    fn collect_mem_ty_sel(
        physdev: &PhysicalDevice,
        dev: &ash::Device,
    ) -> Result<HashMap<MemoryUsage, MemoryTypeFallbackChain>> {
        let nmem_ty = physdev.mem_prop.memory_type_count as usize;
        let mut has_host_visible = 0;
        let mut mem_tys = physdev.mem_prop.memory_types[..nmem_ty]
            .into_iter()
            .enumerate()
            .map(|(mem_ty_idx, mem_ty)| {
                let mem_ty_idx = mem_ty_idx as u32;
                let mem_prop = MemoryProperty(mem_ty.property_flags);
                has_host_visible |= mem_prop.host_visible_bit();
                MemoryType { mem_ty_idx, mem_prop }
            })
            .collect::<Vec<_>>();

        mem_tys.sort_by_key(|mem_ty| mem_ty.mem_prop.dev_score());
        let dev_tys = mem_tys.iter()
            .cloned()
            .collect::<Vec<_>>();

        mem_tys.sort_by_key(|mem_ty| mem_ty.mem_prop.push_score());
        let push_tys = mem_tys.iter()
            .take_while(|mem_ty| mem_ty.mem_prop.host_visible_bit() != 0)
            .cloned()
            .collect::<Vec<_>>();

        mem_tys.sort_by_key(|mem_ty| mem_ty.mem_prop.pull_score());
        let pull_tys = mem_tys.iter()
            .take_while(|mem_ty| mem_ty.mem_prop.host_visible_bit() != 0)
            .cloned()
            .collect::<Vec<_>>();

        if has_host_visible == 0 {
            // There is no host visible memory type. We cannot pass data to the
            // device.
            return Err(Error::UnsupportedPlatform);
        }

        let mut sel = HashMap::new();
        sel.insert(MemoryUsage::Device, MemoryTypeFallbackChain(dev_tys));
        sel.insert(MemoryUsage::Push, MemoryTypeFallbackChain(push_tys));
        sel.insert(MemoryUsage::Pull, MemoryTypeFallbackChain(pull_tys));
        Ok(sel)
    }
    pub fn new(
        physdev: &PhysicalDevice,
        dev: &ash::Device,
        trans_queue: Option<vk::Queue>,
    ) -> Result<DeviceMemoryAllocationDetail> {
        let mem_ty_sel = Self::collect_mem_ty_sel(physdev, dev)?;
        let stage = trans_queue
            .map(|q| MemoryStaging { trans_queue: q, ..Default::default() });
        let pallocs = Default::default();
        let malloc_detail = DeviceMemoryAllocationDetail { mem_ty_sel, stage,
            pallocs };
        Ok(malloc_detail)
    }
}

struct DevicePresentationDetail {
    present_queue: vk::Queue,
    swapchain: vk::SwapchainKHR,
}
impl DevicePresentationDetail {
    const PREFERRED_IMG_COUNT: u32 = 3;
    fn create_swapchain(
        physdev: &PhysicalDevice,
        dev_cap_detail: &DeviceCapabilityDetail,
    ) -> Result<vk::SwapchainKHR> {
        let khr_swapchain = dev_cap_detail.dev_exts.khr_swapchain.as_ref()
            .ok_or(Error::InvalidOperation)?;
        let physdev_surf_detail = physdev.surf_detail.as_ref()
            .ok_or(Error::InvalidOperation)?;
        let ctxt_surf_detail = physdev.ctxt.surf_detail.as_ref()
            .ok_or(Error::InvalidOperation)?;

        let nimg = Self::PREFERRED_IMG_COUNT
            .min(physdev_surf_detail.surf_cap.max_image_count);
        let fmt = physdev_surf_detail.surf_fmt.format;
        let color_space = physdev_surf_detail.surf_fmt.color_space;
        let trans = physdev_surf_detail.surf_cap.current_transform;

        let width = ctxt_surf_detail.width;
        let height = ctxt_surf_detail.height;

        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(ctxt_surf_detail.surf)
            .min_image_count(nimg)
            .image_format(fmt)
            .image_color_space(color_space)
            .image_extent(vk::Extent2D { width, height })
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .present_mode(physdev_surf_detail.present_mode)
            .clipped(true)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .build();

        let swapchain = unsafe {
            khr_swapchain.create_swapchain(&create_info, None)?
        };
        info!("created swapchain");
        Ok(swapchain)
    }
    pub fn new(
        physdev: &PhysicalDevice,
        dev_cap_detail: &DeviceCapabilityDetail,
        present_queue: vk::Queue,
    ) -> Result<DevicePresentationDetail> {
        let swapchain = Self::create_swapchain(physdev, dev_cap_detail)?;
        let present_detail = DevicePresentationDetail { present_queue,
            swapchain };
        Ok(present_detail)
    }
    pub fn wipe(&mut self, dev_cap_detail: &DeviceCapabilityDetail) {
        let khr_swapchain = dev_cap_detail.dev_exts.khr_swapchain.as_ref()
            .unwrap();
        unsafe { khr_swapchain.destroy_swapchain(self.swapchain, None) };
    }
}


struct DeviceGraphicsDetail {
    graph_queue: vk::Queue,
}
impl DeviceGraphicsDetail {
    pub fn new(graph_queue: vk::Queue) -> Result<DeviceGraphicsDetail> {
        let graph_detail = DeviceGraphicsDetail { graph_queue };
        Ok(graph_detail)
    }
}


struct DeviceComputeDetail {
    comp_queue: vk::Queue,
}
impl DeviceComputeDetail {
    pub fn new(comp_queue: vk::Queue) -> Result<DeviceComputeDetail> {
        let comp_detail = DeviceComputeDetail { comp_queue };
        Ok(comp_detail)
    }
}


#[derive(Clone)]
pub struct BufferConfig {
    pub size: usize,
    pub usage: vk::BufferUsageFlags,
}
impl_ptr_wrapper!(Buffer -> BufferInner);
pub struct Buffer(Arc<BufferInner>);
pub struct BufferInner {
    buf: vk::Buffer,
    mem: MemorySlice,
    cfg: BufferConfig,
}
impl Buffer {
    fn create_buf(dev: &Device, buf_cfg: &BufferConfig) -> Result<vk::Buffer> {
        if buf_cfg.usage.as_raw() == 0 { return Err(Error::InvalidOperation) }
        let create_info = vk::BufferCreateInfo::builder()
            .size(buf_cfg.size as u64)
            .usage(buf_cfg.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let buf = unsafe { dev.dev.create_buffer(&create_info, None)? };
        info!("created buffer");
        Ok(buf)
    }
    pub fn new(
        dev: &Device,
        buf_cfg: &BufferConfig,
        mem_usage: MemoryUsage,
    ) -> Result<Buffer> {
        let buf = Self::create_buf(dev, buf_cfg)?;
        let mem = Device::alloc_buf_mem(dev, buf, mem_usage)?;
        let inner = BufferInner { buf, mem, cfg: buf_cfg.clone() };
        Ok(Buffer(Arc::new(inner)))
    }
}
impl Drop for BufferInner {
    fn drop(&mut self) {
        unsafe { self.mem.mem.dev.dev.destroy_buffer(self.buf, None) };
        info!("destroyed image");
    }
}

#[derive(Clone)]
pub struct ImageConfig {
    pub fmt: vk::Format,
    pub view_ty: vk::ImageViewType,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub nlayer: u32,
    pub nmip: u32,
    pub usage: vk::ImageUsageFlags,
}
pub struct ImageInner {
    img: vk::Image,
    img_view: vk::ImageView,
    mem: MemorySlice,
    cfg: ImageConfig,
}
impl_ptr_wrapper!(Image -> ImageInner);
pub struct Image(Arc<ImageInner>);
impl Image {
    fn create_img(dev: &Device, img_cfg: &ImageConfig) -> Result<vk::Image> {
        if img_cfg.usage.as_raw() == 0 { return Err(Error::InvalidOperation) }
        let flags = match img_cfg.view_ty {
            vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY => {
                vk::ImageCreateFlags::CUBE_COMPATIBLE
            },
            _ => vk::ImageCreateFlags::empty(),
        };
        let img_ty = match img_cfg.view_ty {
            vk::ImageViewType::TYPE_1D => vk::ImageType::TYPE_1D,
            vk::ImageViewType::TYPE_2D |
                vk::ImageViewType::CUBE |
                vk::ImageViewType::TYPE_1D_ARRAY => vk::ImageType::TYPE_2D,
            vk::ImageViewType::TYPE_3D |
                vk::ImageViewType::TYPE_2D_ARRAY |
                vk::ImageViewType::CUBE_ARRAY => vk::ImageType::TYPE_2D,
            _ => unreachable!(),
        };
        let extent = vk::Extent3D {
            width: img_cfg.width,
            height: img_cfg.height,
            depth: img_cfg.depth,
        };
        let create_info = vk::ImageCreateInfo::builder()
            .flags(flags)
            .image_type(img_ty)
            .format(img_cfg.fmt)
            .extent(extent)
            .mip_levels(img_cfg.nmip)
            .array_layers(img_cfg.nlayer)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(img_cfg.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let img = unsafe { dev.dev.create_image(&create_info, None)? };
        info!("created image");
        Ok(img)
    }
    fn create_img_view(
        dev: &Device,
        img_cfg: &ImageConfig,
        img: vk::Image,
    ) -> Result<vk::ImageView> {
        let subrsc_rng = vk::ImageSubresourceRange {
            aspect_mask: fmt2aspect(img_cfg.fmt)?,
            base_mip_level: 0,
            level_count: img_cfg.nmip,
            base_array_layer: 0,
            layer_count: img_cfg.nlayer,
        };
        let create_info = vk::ImageViewCreateInfo::builder()
            .image(img)
            .view_type(img_cfg.view_ty)
            .format(img_cfg.fmt)
            .subresource_range(subrsc_rng)
            .build();
        let img_view = unsafe {
            dev.dev.create_image_view(&create_info, None)?
        };
        Ok(img_view)
    }
    pub fn new(
        dev: &Device,
        img_cfg: &ImageConfig,
        mem_use: MemoryUsage,
    ) -> Result<Image> {
        let img = Self::create_img(dev, img_cfg)?;
        let mem = Device::alloc_img_mem(dev, img, mem_use)?;
        let img_view = Self::create_img_view(dev, img_cfg, img)?;
        let inner = ImageInner { img, img_view, mem, cfg: img_cfg.clone() };
        Ok(Image(Arc::new(inner)))
    }
}
impl Drop for ImageInner {
    fn drop(&mut self) {
        unsafe { self.mem.mem.dev.dev.destroy_image(self.img, None) };
        info!("destroyed image");
    }
}


pub struct ShaderModuleInner {
    pub(crate) dev: Device,
    pub shader_mod: vk::ShaderModule,
    pub manifests: Vec<EntryPointManifest>,
}
impl_ptr_wrapper!(ShaderModule -> ShaderModuleInner);
pub struct ShaderModule(Arc<ShaderModuleInner>);
impl Drop for ShaderModuleInner {
    fn drop(&mut self) {
        unsafe { self.dev.dev.destroy_shader_module(self.shader_mod, None) };
        info!("destroyed shader module");
    }
}
impl ShaderModule {
    fn create_shader_mod(
        dev: &Device,
        spv: &SpirvBinary,
    ) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(spv.words())
            .build();
        let shader_mod = unsafe {
            dev.dev.create_shader_module(&create_info, None)?
        };
        info!("created shader module");
        Ok(shader_mod)
    }
    pub fn new(dev: &Device, spv: &SpirvBinary) -> Result<ShaderModule> {
        let dev = dev.clone();
        let shader_mod = Self::create_shader_mod(&dev, spv)?;
        let manifests = spv.reflect()?.to_vec();
        let inner = ShaderModuleInner { dev, shader_mod, manifests };
        Ok(ShaderModule(Arc::new(inner)))
    }
    pub fn entry_points(
        &self
    ) -> impl Iterator<Item=Result<ShaderEntryPoint>> + '_ {
        (0..self.manifests.len()).into_iter()
            .map(move |i| ShaderEntryPoint::new(self, i))
    }
}

// TODO: Separate the inner.
#[derive(Clone)]
pub struct ShaderEntryPoint {
    shader_mod: ShaderModule,
    ientry_point: usize,
}
impl ShaderEntryPoint {
    fn new(
        shader_mod: &ShaderModule,
        ientry_point: usize,
    ) -> Result<ShaderEntryPoint> {
        let shader_mod = shader_mod.clone();
        Ok(ShaderEntryPoint { shader_mod, ientry_point })
    }
    fn name(&self) -> &str {
        &self.manifest().name
    }
    fn stage(&self) -> Option<vk::ShaderStageFlags> {
        use spirq::ExecutionModel::*;
        let stage = match self.manifest().exec_model {
            Vertex => vk::ShaderStageFlags::VERTEX,
            TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
            TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            Geometry => vk::ShaderStageFlags::GEOMETRY,
            Fragment => vk::ShaderStageFlags::FRAGMENT,
            GLCompute => vk::ShaderStageFlags::COMPUTE,
            _ => { return None }
        };
        Some(stage)
    }
    pub fn manifest(&self) -> &EntryPointManifest {
        &self.shader_mod.manifests[self.ientry_point]
    }
}

pub struct ShaderArrayInner {
    entry_points: Vec<ShaderEntryPoint>,
    manifest: Manifest,
    /// `HashMap` for DescriptorUpdateTemplate.
    desc_set_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    desc_pool_sizes: Vec<vk::DescriptorPoolSize>,
    push_const_rng: Vec<vk::PushConstantRange>,
}
impl Drop for ShaderArrayInner {
    fn drop(&mut self) {
        for &desc_set_layout in self.desc_set_layouts.values() {
            let dev = &self.entry_points.first().unwrap().shader_mod.dev;
            unsafe {
                dev.dev.destroy_descriptor_set_layout(desc_set_layout, None);
            }
        }
    }
}
impl_ptr_wrapper!(ShaderArray -> ShaderArrayInner);
pub struct ShaderArray(Arc<ShaderArrayInner>);
impl ShaderArray {
    /// `entry_points` should be sorted externally to represent the actual order
    /// of execution, otherwise the input and output interface would be
    /// corrupted.
    pub fn new(entry_points: &[ShaderEntryPoint]) -> Result<ShaderArray> {
        if entry_points.len() == 0 { return Err(Error::InvalidOperation) }
        let manifest = {
            let mut manifest = Manifest::default();
            for entry_point in entry_points {
                manifest.merge(entry_point.manifest());
            }
            manifest
        };
        let entry_points = entry_points.iter().cloned().collect::<Vec<_>>();
        let dev = &entry_points.first().unwrap().shader_mod.dev;

        let mut pool_size_map = HashMap::<vk::DescriptorType, u32>::new();
        let mut set_bind_map = HashMap::<u32, Vec<vk::DescriptorSetLayoutBinding>>::new();
        for desc in manifest.descs() {
            let (set, bind) = desc.desc_bind.into_inner();
            // Process actual descriptor bindings.
            let nbind = desc.desc_ty.nbind();
            let desc_ty = match desc.desc_ty {
                DescriptorType::UniformBuffer(..) => {
                    vk::DescriptorType::UNIFORM_BUFFER
                },
                DescriptorType::StorageBuffer(..) => {
                    vk::DescriptorType::STORAGE_BUFFER
                },
                DescriptorType::Image(..) => {
                    vk::DescriptorType::SAMPLED_IMAGE
                },
                DescriptorType::Sampler(..) => {
                    vk::DescriptorType::SAMPLER
                }
                DescriptorType::SampledImage(..) => {
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                }
                DescriptorType::InputAttachment(..) => {
                    vk::DescriptorType::INPUT_ATTACHMENT
                },
            };
            let dslb = vk::DescriptorSetLayoutBinding::builder()
                .binding(bind)
                .descriptor_type(desc_ty)
                .descriptor_count(nbind)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build();
            set_bind_map.entry(set).or_default().push(dslb);
            *pool_size_map.entry(desc_ty).or_default() += 1;
        }
        let push_const_rng = manifest.get_push_const()
            .and_then(|x| x.nbyte())
            .map(|x| {
                vec![
                    vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::ALL,
                        offset: 0,
                        size: x as u32,
                    }
                ]
            })
            .unwrap_or_default();
        let desc_set_layouts = {
            let mut desc_set_layouts = HashMap::with_capacity(set_bind_map.len());
            for (set, dslbs) in set_bind_map {
                let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&dslbs)
                    .build();
                let desc_set_layout = unsafe {
                    dev.dev.create_descriptor_set_layout(&create_info, None)?
                };
                desc_set_layouts.insert(set, desc_set_layout);
            }
            desc_set_layouts
        };
        let desc_pool_sizes = {
            let mut desc_pool_sizes = Vec::with_capacity(pool_size_map.len());
            for (desc_ty, ndesc) in pool_size_map {
                let dps = vk::DescriptorPoolSize {
                    ty: desc_ty,
                    descriptor_count: ndesc,
                };
                desc_pool_sizes.push(dps);
            }
            desc_pool_sizes
        };

        let inner = ShaderArrayInner {
            entry_points, manifest, desc_set_layouts, desc_pool_sizes,
            push_const_rng
        };
        Ok(ShaderArray(Arc::new(inner)))
    }
}





















#[derive(Clone)]
pub struct AttributeBinding {
    /// Vertex input binding.
    pub bind: u32,
    /// Offset from the bound vertex buffer.
    pub offset: usize,
    /// Stride to iterate the bound vertex buffer.
    pub stride: usize,
    /// Format of attribute units.
    pub fmt: vk::Format,
}
pub trait VertexHead {
    /// Get the binding point of a attribute.
    fn attr_bind(&self, loc: u32) -> Option<&AttributeBinding>;
}

#[derive(Clone)]
pub struct AttachmentReference {
    /// The index of attachment in the framebuffer. For shaders using input
    /// attachments, the input attachment indices MUST point to the indices of
    /// corresponding attachments in the framebuffer.
    pub attm_idx: u32,
    /// Format of attachment units. It will be deduced here whether the
    /// attachment is a depth-stencil attachment.
    pub fmt: vk::Format,
    /// The operation done on loading.
    pub load_op: vk::AttachmentLoadOp,
    /// The operation done on storing.
    pub store_op: vk::AttachmentStoreOp,
    /// Blend state of the attachment. Any non-`None` value will enable
    /// attachment blending.
    pub blend_state: Option<vk::PipelineColorBlendAttachmentState>,
    /// Final layout after attachment write.
    pub final_layout: vk::ImageLayout,
}
pub trait FragmentHead {
    fn attm_ref(&self, loc: u32) -> Option<&AttachmentReference>;
    fn depth_attm_idx(&self, depth_attm_id: u32) -> Option<u32>;
}

#[derive(Default)]
pub struct GraphicsDepthStencilConfig {
    /// Compare operator used in depth test.
    cmp_op: vk::CompareOp,
    /// Any non-`None` value in this field will enable depth bounds testing.
    /// Primitives are only drawn if they falls in this range of depth, where
    /// the first value if the minimal allowed depth and the second the maximum.
    depth_range: Option<(f32, f32)>,
    /// Any non-`None` value in this field will enable stencil testing. The
    /// first value is the front op state, and the second the back op state.
    stencil_ops: Option<(vk::StencilOpState, vk::StencilOpState)>,
}
#[derive(Default)]
pub struct GraphicsRasterizationConfig {
    pub wireframe: bool,
    pub cull_mode: vk::CullModeFlags,
}
#[derive(Default)]
pub struct GraphicsColorBlendConfig {
    pub src_factor: vk::BlendFactor,
    pub dst_factor: vk::BlendFactor,
    pub blend_op: vk::BlendOp,
    pub blend_consts: [f32; 4],
}
pub struct GraphicsPipeline {
    pub shader_arr: ShaderArray,
    pub attr_map: HashMap<InterfaceLocation, AttributeBinding>,
    pub attm_map: HashMap<InterfaceLocation, AttachmentReference>,
    /// Index of the depth-stencil attachment to be written in the framebuffer.
    pub depth_attm_idx: Option<u32>,
    pub raster_cfg: GraphicsRasterizationConfig,
    /// Any non-`None` value will enable depth testing.
    pub depth_cfg: Option<GraphicsDepthStencilConfig>,
    /// Any non-`None` value will enable alpha blending. All attachments will
    /// share the same blending configuration.
    pub blend_cfg: Option<GraphicsColorBlendConfig>,
}
impl GraphicsPipeline {
    /// Note that `depth_attm_id` is NOT the index of attachment in framebuffer
    /// but identifier to be used to query the provided `frag_head`.
    pub fn new<VH: VertexHead, FH: FragmentHead>(
        shader_arr: &ShaderArray,
        vert_head: &VH,
        frag_head: &FH,
        depth_attm_id: Option<u32>,
        raster_cfg: GraphicsRasterizationConfig,
        depth_cfg: Option<GraphicsDepthStencilConfig>,
        blend_cfg: Option<GraphicsColorBlendConfig>,
    ) -> Result<GraphicsPipeline> {
        // Ensure the vertex shader is the first and the fragment shader the
        // last.
        let manifest = &shader_arr.manifest;
        let attr_map = Self::collect_attr_map(manifest, vert_head)?;
        let attm_map = Self::collect_attm_map(manifest, frag_head)?;
        let depth_attm_idx = if let Some(depth_attm_id) = depth_attm_id {
            let depth_attm_idx = frag_head.depth_attm_idx(depth_attm_id)
                .ok_or(Error::InvalidOperation)?;
            Some(depth_attm_idx)
        } else { None };
        let shader_arr = shader_arr.clone();
        let graph_pipe_req = GraphicsPipeline {
            shader_arr, attr_map, attm_map, depth_attm_idx, raster_cfg,
            depth_cfg, blend_cfg
        };
        Ok(graph_pipe_req)
    }
    fn collect_attr_map(
        manifest: &Manifest,
        vert_head: &dyn VertexHead,
    ) -> Result<HashMap<InterfaceLocation, AttributeBinding>> {
        use std::collections::hash_map::Entry::Vacant;
        let mut attr_map = HashMap::new();
        for attr_res in manifest.inputs() {
            let location = attr_res.location;
            if let None = attr_res.ty.nbyte() {
                return Err(Error::PipelineMismatched("attachment cannot be opaque type"));
            }
            let attr_bind = vert_head.attr_bind(location.loc())
                // Attribute not supported.
                .ok_or(Error::PipelineMismatched("attribute not supported by head"))?;
            if let Vacant(entry) = attr_map.entry(location) {
                entry.insert(attr_bind.clone());
            } else {
                return Err(Error::PipelineMismatched("attribute location collision"));
            }
        }
        Ok(attr_map)
    }
    fn collect_attm_map(
        manifest: &Manifest,
        frag_head: &dyn FragmentHead,
    ) -> Result<HashMap<InterfaceLocation, AttachmentReference>> {
        use std::collections::hash_map::Entry::Vacant;
        let mut attm_map = HashMap::new();
        for attm_ref in manifest.outputs() {
            let location = attm_ref.location;
            if let None = attm_ref.ty.nbyte() {
                return Err(Error::PipelineMismatched("attachment cannot be opaque type"));
            }
            let attm_ref = frag_head.attm_ref(location.loc())
                .ok_or(Error::PipelineMismatched("attachment not supported by head"))?;
            if let Vacant(entry) = attm_map.entry(location) {
                entry.insert(attm_ref.clone());
            } else {
                return Err(Error::PipelineMismatched("attachment location collision"));
            }
        }
        Ok(attm_map)
    }
}


fn is_depth_fmt(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::D16_UNORM |
        vk::Format::X8_D24_UNORM_PACK32 |
        vk::Format::D32_SFLOAT |
        vk::Format::S8_UINT |
        vk::Format::D16_UNORM_S8_UINT |
        vk::Format::D24_UNORM_S8_UINT |
        vk::Format::D32_SFLOAT_S8_UINT => true,
        _ => false,
    }
}


struct PipelineInner {
    dev: Device,
    pipe: vk::Pipeline,
    pipe_layout: vk::PipelineLayout,
}
impl Drop for PipelineInner {
    fn drop(&mut self) {
        unsafe {
            self.dev.dev.destroy_pipeline(self.pipe, None);
            self.dev.dev.destroy_pipeline_layout(self.pipe_layout, None);
        }
    }
}
impl_ptr_wrapper!(Pipeline -> PipelineInner);
struct Pipeline(Arc<PipelineInner>);
impl Pipeline {
    fn new(
        dev: &Device,
        pipe: vk::Pipeline,
        pipe_layout: vk::PipelineLayout,
    ) -> Pipeline {
        let dev = dev.clone();
        let inner = PipelineInner { dev, pipe, pipe_layout };
        Pipeline(Arc::new(inner))
    }
}

pub struct RenderPassInner {
    dev: Arc<DeviceInner>,
    pass: vk::RenderPass,
    framebuf: vk::Framebuffer,
    pipes: Vec<Pipeline>,
    imgs: Vec<Image>, // Render targets.
}
impl Drop for RenderPassInner {
    fn drop(&mut self) {
        unsafe {
            self.dev.dev.destroy_render_pass(self.pass, None);
            self.dev.dev.destroy_framebuffer(self.framebuf, None);
        }
    }
}
impl_ptr_wrapper!(RenderPass -> RenderPassInner);
pub struct RenderPass(Arc<RenderPassInner>);
impl RenderPass {
    fn match_img_extent(render_targets: &[Image]) -> Result<(u32, u32)> {
        // A render pass is allowed to have no attachment but we don't consider
        // such scenario. Error will be raised if there is no render target at
        // all.
        let mut size = None;
        for img in render_targets.iter() {
            let cur_width = img.cfg.width;
            let cur_height = img.cfg.height;
            if let Some((width, height)) = size {
                if (width != cur_width) || (height != cur_height) {
                    return Err(Error::PipelineMismatched(
                        "render target size mismatched"));
                }
            } else {
                size = Some((cur_width, cur_height));
            }
        }
        size.ok_or(Error::InvalidOperation)
    }
    /// `graph_pipes` should be sorted in order to represent the command
    /// recording order.
    pub fn new(
        dev: &Device,
        graph_pipes: &[GraphicsPipeline],
        dependencies: &[vk::SubpassDependency],
        render_targets: &[Image],
    ) -> Result<RenderPass> {
        let (width, height) = Self::match_img_extent(render_targets)?;
        let pass = {
            let attm_ref_map = graph_pipes.iter()
                .flat_map(|graph_pipe| {
                    graph_pipe.attm_map.values()
                        .map(|attm_ref| (attm_ref.attm_idx, attm_ref))
                })
                .collect::<HashMap<_, _>>();
            let attm_descs = attm_ref_map.iter()
                .map(|(_attm_idx, attm_ref)| {
                    vk::AttachmentDescription {
                        flags: Default::default(),
                        format: attm_ref.fmt,
                        samples: vk::SampleCountFlags::TYPE_1,
                        load_op: attm_ref.load_op,
                        store_op: attm_ref.store_op,
                        stencil_load_op: attm_ref.load_op,
                        stencil_store_op: attm_ref.store_op,
                        initial_layout: vk::ImageLayout::UNDEFINED,
                        final_layout: attm_ref.final_layout,
                    }
                })
                .collect::<Vec<_>>();
            struct SubpassDescriptionDetail {
                input_attms: Vec<vk::AttachmentReference>,
                color_attms: Vec<vk::AttachmentReference>,
                depth_attm: Option<vk::AttachmentReference>,
            }
            let mut subpass_desc_details = Vec::with_capacity(graph_pipes.len());
            for graph_pipe in graph_pipes {
                let mut input_attms = Vec::new();
                for desc_res in graph_pipe.shader_arr.manifest.descs() {
                    if let DescriptorType::InputAttachment(nbind, i) = desc_res.desc_ty {
                        let pad_end = *i as usize;
                        if input_attms.len() < pad_end {
                            let stuff = (input_attms.len()..pad_end).into_iter()
                                .map(|_| {
                                    vk::AttachmentReference {
                                        attachment: vk::ATTACHMENT_UNUSED,
                                        layout: Default::default(),
                                    }
                                });
                            input_attms.extend(stuff);
                        }
                        let attm_fmt = attm_ref_map.get(&i)
                            .ok_or(Error::InvalidOperation)?
                            .fmt;
                        let layout = if is_depth_fmt(attm_fmt) {
                            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                        } else {
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                        };
                        let attm_ref = vk::AttachmentReference {
                            attachment: *i,
                            layout
                        };
                        let attm_refs = std::iter::repeat(attm_ref)
                            .take(*nbind as usize);
                        input_attms.extend(attm_refs);
                    }
                }
                let mut color_attms = Vec::new();
                for (location, attm_ref) in graph_pipe.attm_map.iter() {
                    let loc = location.loc() as usize;
                    if color_attms.len() < loc {
                        let stuff = (color_attms.len()..loc).into_iter()
                            .map(|_| {
                                vk::AttachmentReference {
                                    attachment: vk::ATTACHMENT_UNUSED,
                                    layout: Default::default(),
                                }
                            });
                        color_attms.extend(stuff);
                    }
                    let attm_ref = vk::AttachmentReference {
                        attachment: attm_ref.attm_idx,
                        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    };
                    color_attms.push(attm_ref);
                }
                let depth_attm = graph_pipe.depth_attm_idx
                    .map(|depth_attm_idx| {
                        vk::AttachmentReference {
                            attachment: depth_attm_idx,
                            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                        }
                    });
                let subpass_desc_detail = SubpassDescriptionDetail {
                    color_attms, input_attms, depth_attm
                };
                subpass_desc_details.push(subpass_desc_detail);
            };
            
            let subpass_descs = subpass_desc_details.iter()
                .map(|subpass_desc_detail| {
                    let mut x = vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&subpass_desc_detail.color_attms)
                        .input_attachments(&subpass_desc_detail.color_attms);
                    if let Some(ref depth_attm) = subpass_desc_detail.depth_attm {
                        x = x.depth_stencil_attachment(depth_attm);
                    }
                    x.build()
                })
                .collect::<Vec<_>>();

            let create_info = vk::RenderPassCreateInfo::builder()
                .attachments(&attm_descs)
                .subpasses(&subpass_descs)
                .dependencies(dependencies)
                .build();
            let pass = unsafe {
                dev.0.dev.create_render_pass(&create_info, None)?
            };
            pass
        };
        let pipes = {
            struct GraphicsPipelineDescriptionDetail {
                entry_names: Vec<std::ffi::CString>,
                psscis: Vec<vk::PipelineShaderStageCreateInfo>,
                vert_binds: Vec<vk::VertexInputBindingDescription>,
                vert_attrs: Vec<vk::VertexInputAttributeDescription>,
                pvisci: vk::PipelineVertexInputStateCreateInfo,
                piasci: vk::PipelineInputAssemblyStateCreateInfo,
                ptsci: vk::PipelineTessellationStateCreateInfo,
                pvsci: vk::PipelineViewportStateCreateInfo,
                prsci: vk::PipelineRasterizationStateCreateInfo,
                pmsci: vk::PipelineMultisampleStateCreateInfo,
                pdssci: vk::PipelineDepthStencilStateCreateInfo,
                attm_blends: Vec<vk::PipelineColorBlendAttachmentState>,
                pcbsci: vk::PipelineColorBlendStateCreateInfo,
                pdsci: vk::PipelineDynamicStateCreateInfo,
                pipe_layout: vk::PipelineLayout,
            }

            let viewports = &[
                vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: width as f32,
                    height: height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                },
            ];
            let scissors = &[
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width: width, height: height },
                },
            ];

            let mut graph_pipe_desc_details = Vec::new();
            for graph_pipe in graph_pipes {
                let entry_names = graph_pipe.shader_arr.entry_points.iter()
                    .map(|entry_point| {
                        std::ffi::CString::new(entry_point.name())
                            .expect("invalid entry point name")
                    })
                    .collect::<Vec<_>>();
                let psscis = graph_pipe.shader_arr.entry_points.iter()
                    .enumerate()
                    .map(|(i, entry_point)| {
                        vk::PipelineShaderStageCreateInfo::builder()
                            .stage(entry_point.stage().expect("invalid stage"))
                            .module(entry_point.shader_mod.shader_mod)
                            // .specialization_info(/* ... */) // TODO
                            .name(&entry_names[i])
                            .build()
                    })
                    .collect::<Vec<_>>();
                let mut vert_bind_map = HashMap::new();
                let mut vert_attr_map = HashMap::new();
                for (location, attr_bind) in graph_pipe.attr_map.iter() {
                    use std::collections::hash_map::Entry::Vacant;
                    if let Vacant(entry) = vert_bind_map.entry(attr_bind.bind) {
                        let vert_bind = vk::VertexInputBindingDescription {
                            binding: attr_bind.bind,
                            stride: attr_bind.stride as u32,
                            input_rate: vk::VertexInputRate::VERTEX,
                        };
                        entry.insert(vert_bind);
                    }
                    if let Vacant(entry) = vert_attr_map.entry(*location) {
                        let vert_attr = vk::VertexInputAttributeDescription {
                            location: location.loc(),
                            binding: attr_bind.bind,
                            format: attr_bind.fmt,
                            offset: attr_bind.offset as u32,
                        };
                        entry.insert(vert_attr);
                    }
                }
                let vert_binds = vert_bind_map.into_iter()
                    .map(|x| x.1)
                    .collect::<Vec<_>>();
                let vert_attrs = vert_attr_map.into_iter()
                    .map(|x| x.1)
                    .collect::<Vec<_>>();
                let pvisci = vk::PipelineVertexInputStateCreateInfo::builder()
                    .vertex_binding_descriptions(&vert_binds)
                    .vertex_attribute_descriptions(&vert_attrs)
                    .build();

                
                let piasci = vk::PipelineInputAssemblyStateCreateInfo::builder()
                    .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                    // TODO: This one is interesting. Wanna look into it?
                    .primitive_restart_enable(false)
                    .build();

                // NOTE: Don't use tesselation and geometry shaders because
                // these stages will generate unknown number of primitives,
                // which will disable tile-based rendering which has been used
                // broadly.
                let ptsci = vk::PipelineTessellationStateCreateInfo::builder()
                    .build();

                let pvsci = vk::PipelineViewportStateCreateInfo::builder()
                    .viewport_count(viewports.len() as u32)
                    .viewports(viewports)
                    .scissor_count(scissors.len() as u32)
                    .scissors(scissors)
                    .build();

                let poly_mode = if graph_pipe.raster_cfg.wireframe {
                    vk::PolygonMode::LINE
                } else {
                    vk::PolygonMode::FILL
                };
                let prsci = vk::PipelineRasterizationStateCreateInfo::builder()
                    .depth_clamp_enable(false)
                    .rasterizer_discard_enable(false)
                    .polygon_mode(poly_mode)
                    .cull_mode(graph_pipe.raster_cfg.cull_mode)
                    .depth_bias_enable(false)
                    .line_width(1.0)
                    .build();

                let pmsci = vk::PipelineMultisampleStateCreateInfo::builder()
                    .rasterization_samples(vk::SampleCountFlags::TYPE_1)
                    .sample_shading_enable(false)
                    .min_sample_shading(1.0)
                    .build();

                let mut pdssci = vk::PipelineDepthStencilStateCreateInfo::builder()
                    .build();
                if let Some(depth_cfg) = graph_pipe.depth_cfg.as_ref() {
                    pdssci.depth_test_enable = 1;
                    pdssci.depth_compare_op = depth_cfg.cmp_op;
                    // This will be ignored if there is no depth-stencil
                    // attachment attached.
                    pdssci.depth_write_enable = 1;
                    if dev.cap_detail.feats.depth_bounds != 0 {
                        if let Some((min_depth, max_depth)) = depth_cfg.depth_range.as_ref() {
                            pdssci.depth_bounds_test_enable = 1;
                            pdssci.min_depth_bounds = *min_depth;
                            pdssci.max_depth_bounds = *max_depth;
                        }
                    }
                    if let Some((front_op, back_op)) = depth_cfg.stencil_ops.as_ref() {
                        pdssci.stencil_test_enable = 1;
                        pdssci.front = *front_op;
                        pdssci.back = *back_op;
                    }
                }

                let mut blend_consts = Default::default();
                let mut attm_blends = Vec::new();
                if let Some(blend_cfg) = graph_pipe.blend_cfg.as_ref() {
                    let attm_blend = vk::PipelineColorBlendAttachmentState::builder()
                        .blend_enable(true)
                        .src_color_blend_factor(blend_cfg.src_factor)
                        .dst_color_blend_factor(blend_cfg.dst_factor)
                        .color_blend_op(blend_cfg.blend_op)
                        .src_alpha_blend_factor(blend_cfg.src_factor)
                        .dst_alpha_blend_factor(blend_cfg.dst_factor)
                        .alpha_blend_op(blend_cfg.blend_op)
                        .color_write_mask(vk::ColorComponentFlags::all())
                        .build();
                    attm_blends = std::iter::repeat(attm_blend)
                        .take(graph_pipe.attm_map.len())
                        .collect::<Vec<_>>();
                    blend_consts = blend_cfg.blend_consts
                }
                let pcbsci = vk::PipelineColorBlendStateCreateInfo::builder()
                    .attachments(&attm_blends)
                    .blend_constants(blend_consts)
                    .build();

                let pdsci = vk::PipelineDynamicStateCreateInfo::builder()
                    .build();

                let desc_set_layouts = &graph_pipe.shader_arr.desc_set_layouts;
                let desc_set_layouts = desc_set_layouts.values()
                    .copied()
                    .collect::<Vec<_>>();
                let create_info = vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&desc_set_layouts)
                    .push_constant_ranges(&graph_pipe.shader_arr.push_const_rng)
                    .build();
                let pipe_layout = unsafe {
                    dev.0.dev.create_pipeline_layout(&create_info, None)?
                };

                let graph_pipe_desc_detail = GraphicsPipelineDescriptionDetail {
                    entry_names, psscis, vert_binds, vert_attrs, pvisci, piasci,
                    ptsci, pvsci, prsci, pmsci, pdssci, attm_blends, pcbsci,
                    pdsci, pipe_layout
                };
                graph_pipe_desc_details.push(graph_pipe_desc_detail);
            }

            let create_infos = graph_pipe_desc_details.iter()
                .enumerate()
                .map(|(i, graph_pipe_desc_detail)| {
                    vk::GraphicsPipelineCreateInfo::builder()
                        .stages(&graph_pipe_desc_detail.psscis)
                        .vertex_input_state(&graph_pipe_desc_detail.pvisci)
                        .input_assembly_state(&graph_pipe_desc_detail.piasci)
                        .tessellation_state(&graph_pipe_desc_detail.ptsci)
                        .viewport_state(&graph_pipe_desc_detail.pvsci)
                        .rasterization_state(&graph_pipe_desc_detail.prsci)
                        .multisample_state(&graph_pipe_desc_detail.pmsci)
                        .depth_stencil_state(&graph_pipe_desc_detail.pdssci)
                        .color_blend_state(&graph_pipe_desc_detail.pcbsci)
                        .dynamic_state(&graph_pipe_desc_detail.pdsci)
                        .layout(graph_pipe_desc_detail.pipe_layout)
                        .render_pass(pass)
                        .subpass(i as u32)
                        .build()
                })
                .collect::<Vec<_>>();
            let pipe_cache = vk::PipelineCache::null();
            let pipes = unsafe {
                dev.0.dev.create_graphics_pipelines(pipe_cache, &create_infos, None)
            };
            let pipes = match pipes {
                Ok(pipes) => pipes,
                Err((pipes, e)) => {
                    for pipe in pipes {
                        if pipe != vk::Pipeline::null() {
                            unsafe { dev.0.dev.destroy_pipeline(pipe, None) };
                        }
                    }
                    return Err(e.into());
                }
            };
            let pipes = pipes.into_iter()
                .enumerate()
                .map(|(i, pipe)| {
                    let pipe_layout = graph_pipe_desc_details[i].pipe_layout;
                    Pipeline::new(&dev, pipe, pipe_layout)
                })
                .collect::<Vec<_>>();
            pipes
        };
        let framebuf = {
            let attms = render_targets.iter()
                .map(|x| x.img_view)
                .collect::<Vec<_>>();
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(pass)
                .attachments(&attms)
                .width(width)
                .height(height)
                .layers(1) // TODO: Support multilayer rendering in the future.
                .build();
            let framebuf = unsafe {
                dev.0.dev.create_framebuffer(&create_info, None)?
            };
            framebuf
        };
        let imgs = render_targets.to_owned();
        let dev = dev.0.clone();

        let inner = RenderPassInner { dev, pass, framebuf, pipes, imgs };
        Ok(RenderPass(Arc::new(inner)))
    }
}

struct Presentation {
    dev: Arc<DeviceInner>,
}






#[derive(Clone)]
pub enum BindPoint {
    /// Bind a resource to a descriptor binding point. Descriptor set will be
    /// internally managed by `FlowState`. Descriptor's type will be
    /// automatically inferred shader module metadata. The binding will be
    /// visible to all following `draw` calls.
    ///
    /// NOTE: Bound variable MUST be `Buffer` or `Image`.
    Descriptor(u32, u32),
    /// Bind a vertex buffer for graphics pipeline. The binding will be visible
    /// to all following `draw` calls.
    ///
    /// NOTE: Bound variable must be `Buffer`.
    VertexInput(u32),
    /// Bind an index buffer for graphics pipeline. The binding will be visible
    /// to all following `draw` calls.
    ///
    /// NOTE: Bound variable must be `Buffer`.
    Index,
    /// Bind an color attachment to the framebuffer and bind the attachment to
    /// the pipeline at `location`.
    ///
    /// NOTE: Bound variable MUST be `Image`.
    ColorAttachment(u32),
    /// Insert an depth-stencil attachment to the framebuffer and bind the
    /// attachment to the pipeline.
    ///
    /// NOTE: Bound variable MUST be `Image`.
    DepthAttachment,
}

type VariableToken = usize;
#[derive(Clone)]
struct TransferEventArgs {
    src_var_idx: usize,
    dst_var_idx: usize,
}
#[derive(Clone)]
struct PushConstantEventArgs {
    push_const_var_idx: usize,
}
#[derive(Clone)]
struct BindEventArgs {
    bp: BindPoint,
    var_idx: Option<usize>,
}
#[derive(Clone)]
struct DispatchEventArgs {
    nworkgrp_var_idx: usize,
}
#[derive(Clone)]
struct DrawEventArgs {
    pass: Arc<RenderPassInner>,
    nvert_var_idx: usize,
    ninst_var_idx: usize,
}
#[derive(Clone)]
struct PresentEventArgs {
    // Reserved.
}
#[derive(Clone)]
enum Event {
    Transfer(TransferEventArgs),
    PushConstant(PushConstantEventArgs),
    Bind(BindEventArgs),
    Dispatch(DispatchEventArgs),
    Draw(DrawEventArgs),
    Present(PresentEventArgs),
}
impl Event {
    /// Get the pipeline binding point of the current event. For example,
    /// compute dispatches need to be bound to compute pipeline; but transfers
    /// doesn't need a pipeline so it has no binding point.
    fn pipe_bind(&self) -> Option<vk::PipelineBindPoint> {
        match self {
            Self::Dispatch(_) => Some(vk::PipelineBindPoint::COMPUTE),
            Self::Draw(_) => Some(vk::PipelineBindPoint::GRAPHICS),
            _ => None,
        }
    }
}


struct Chunk {
    /// Serials of dependencies. Events are not triggered *until* the dependency
    /// has finished.
    ///
    /// This dependency is `FlowHead` dependency, not `Chunk` dependency. This
    /// `FlowHead` dependency will be then converted to `Chunk` dependency
    /// later.
    deps: Vec<usize>,
    events: Vec<Event>,
}
pub struct FlowHead {
    serial: usize,
}
pub struct Flow {
    serial: usize,
    chunks: Vec<Chunk>,
}
impl Flow {
    fn new(serial: usize) -> Self {
        let chunk = Chunk {
            deps: Vec::new(),
            events: Vec::new(),
        };
        let chunks = vec![chunk];
        Flow { serial, chunks }
    }


    fn last_chunk(&mut self) -> &mut Chunk {
        if let Some(chunk) = self.chunks.last_mut() {
            return chunk;
        } else {
            unreachable!("flow head always have one chunk ready");
        }
    }
    /// Wait on an previous flow head. Waiting on a flow head itself is
    /// considered no-op.
    pub fn wait_on(&mut self, head: &FlowHead) -> &mut Self {
        // Waiting on self is no-op.
        if head.serial == self.serial {
            return self;
        }
        let last_chunk = self.last_chunk();
        if last_chunk.events.is_empty() {
            // There is no event triggered since last wait, so the new wait can
            // be batched with the previous one.
            last_chunk.deps.push(head.serial);
        } else {
            // There are events triggered before, so we split and add a new
            // chunk.
            let chunk = Chunk {
                deps: vec![head.serial],
                events: Vec::new(),
            };
            self.chunks.push(chunk);
        }
        self
    }
    
    fn push_event(&mut self, event: Event) -> &mut Self {
        self.last_chunk().events.push(event);
        self
    }
    /// Copy the data contained in `src` to `dst`. Staging buffer might be
    /// lazily allocated for inter-chip data transfer if either of the memory is
    /// not host visible. Due to pixel packing, images are always transfered via
    /// a staging buffer.
    ///
    /// NOTE: `src` and `dst` MUST be `HostMemory`, `Buffer` or `Image`.
    pub fn transfer(&mut self, src: VariableToken, dst: VariableToken) -> &mut Self {
        let event = Event::Transfer(TransferEventArgs {
            src_var_idx: src,
            dst_var_idx: dst,
        });
        self.push_event(event);
        self
    }
    /// Push constant data as a part of the control flow to the device.
    ///
    /// NOTE: `push_const` MUST be `HostMemory`.
    pub fn push_const(&mut self, push_const: VariableToken) -> &mut Self {
        let event = Event::PushConstant(PushConstantEventArgs {
            push_const_var_idx: push_const,
        });
        self.push_event(event);
        self
    }
    /// Bind a variable to the binding point.
    ///
    /// NOTE: `var` MUST conform the requirements as described in `BindPoint`
    /// documentation.
    pub fn bind(&mut self, bp: BindPoint, var: Option<VariableToken>) -> &mut Self {
        let event = Event::Bind(BindEventArgs { bp, var_idx: var });
        self.push_event(event);
        self
    }
    /// Dispatch `nworkgrp` workgroups, factorized to optimal dimensions.
    ///
    /// NOTE: `nworkgrp` MUST be `Count`.
    // fn dispatch(&mut self, comp_pipe: &ComputePipeline, nworkgrp: VariableToken) -> &mut Self;
    /// If there is index buffer bound to the pipeline, vertex elements are
    /// indexed by the indices specified in the index buffer.
    pub fn draw(&mut self, pass: &RenderPass, nvert: VariableToken, ninst: VariableToken) -> &mut Self {
        let event = Event::Draw(DrawEventArgs {
            pass: pass.0.clone(),
            nvert_var_idx: nvert,
            ninst_var_idx: ninst,
        });
        self.push_event(event);
        self
    }
    /// Present the current swapchain image.
    pub fn present(&mut self) -> &mut Self {
        let event = Event::Present(PresentEventArgs {});
        self.push_event(event);
        self
    }


    pub fn pause(&self) -> FlowHead {
        FlowHead { serial: self.serial }
    }
}
pub struct FlowGraph {
    /// Chunks of events ordered in a way that all chunks will not depend on any
    /// chunk which has higher index than it.
    chunks: Vec<Chunk>,
    /// Reversed dependency mapping from dependee to dependers.
    rev_dep_map: Vec<Vec<usize>>,
}
impl FlowGraph {
    fn new(head: &FlowHead, flows: &[Flow]) -> FlowGraph {
        let chunks = Self::flatten_flow_refs(head, flows);
        let rev_dep_map = Self::rev_deps(&chunks);
        FlowGraph { chunks, rev_dep_map }
    }
    fn flatten_flow_refs(
        head: &FlowHead,
        flows: &[Flow],
    ) -> Vec<Chunk> {
        fn fn_impl<'a>(
            flow_serial: usize,
            flows: &[Flow],
            chunks: &mut Vec<Chunk>,
            // Mapping from flow serial to a range of `chunks`.
            rng_map: &mut [Option<Range<usize>>],
        ) {
            let beg = chunks.len();
            for (i, chunk) in flows[flow_serial].chunks.iter().enumerate() {
                // Recursively resolve dependency, ensure chunks are already
                // appended before we refer to them.
                for &dep in chunk.deps.iter() {
                    if let None = rng_map[dep] {
                        fn_impl(flow_serial, flows, chunks, rng_map);
                    }
                }
                // Don't replace the `i` below with `chunks.len()`. Remember
                // that `chunks` has been modified.
                let nexdep = chunk.deps.len();
                let mut deps = if i == 0 {
                    // No internal reference for the first chunk.
                    Vec::with_capacity(nexdep)
                } else {
                    // Add internal reference first.
                    let mut deps = Vec::with_capacity(nexdep + 1);
                    deps.push(beg + i - 1);
                    deps
                };
                // Append external reference. Convert flow dependency to chunk
                // dependency.
                let exref = chunk.deps.iter()
                    .map(|&flow_serial| {
                        if let Some(rng) = &rng_map[flow_serial] {
                            rng.end - 1
                        } else { unreachable!() }
                    });
                deps.extend(exref);
    
                let events = chunk.events.clone();
                chunks.push(Chunk { deps, events });
            }
            let end = chunks.len();
            rng_map[flow_serial] = Some(beg..end);
        }
        // All chunks flatted from head.
        let mut chunks = Vec::new();
        // The range of chunks included in a `Flow` in `chunks`.
        let mut rng_map = std::iter::repeat(None)
            .take(flows.len())
            .collect::<Vec<_>>();

        fn_impl(head.serial, &flows, &mut chunks, &mut rng_map);
        // We won't need the flow head.
        chunks
    }
    fn rev_deps(chunks: &[Chunk]) -> Vec<Vec<usize>> {
        // Revert the dependency relation.
        // Mapping from chunk index to the indices of whom depends on the chunk.
        let mut rev_dep_map = std::iter::repeat(Vec::new())
            .take(chunks.len())
            .collect::<Vec<_>>();
        for (i, chunk) in chunks.iter().enumerate() {
            for dep in chunk.deps.iter() {
                rev_dep_map[*dep].push(i);
            }
        }
        rev_dep_map
    }
}


#[derive(Clone, Copy)]
enum VariableType {
    HostMemory,
    Buffer,
    Image,
    Sampler,
    Count,
}

pub struct SymbolSource {
    /// Flows.
    flows: Vec<Flow>,
    /// Variable tokens are mapped to indices of `syms`.
    syms: Vec<VariableType>,
    /// Mapping from symbol names to symbol tokens.
    name_map: HashMap<String, VariableToken>,
}
impl SymbolSource {
    fn new() -> SymbolSource {
        SymbolSource {
            flows: Vec::new(),
            syms: Vec::new(),
            name_map: HashMap::new(),
        }
    }

    pub fn flow(&mut self) -> &mut Flow {
        let serial = self.flows.len();
        self.flows.push(Flow::new(serial));
        &mut self.flows[serial]
    }

    fn push_sym(&mut self, name: &str, ty: VariableType) -> VariableToken {
        let token = self.syms.len();
        self.syms.push(ty);
        self.name_map.insert(name.to_owned(), token);
        token
    }
    /// Declare a piece of unstructured host memory. Push constants and memory
    /// transfer from host will need this type of variables.
    pub fn host_mem(&mut self, name: &str) -> VariableToken {
        self.push_sym(name, VariableType::HostMemory)
    }
    /// Declare a buffer variable.
    pub fn buf(&mut self, name: &str) -> VariableToken {
        self.push_sym(name, VariableType::Buffer)
    }
    /// Declare a image variable.
    pub fn img(&mut self, name: &str) -> VariableToken {
        self.push_sym(name, VariableType::Image)
    }
    /// Declare a separable sampler.
    pub fn sampler(&mut self, name: &str) -> VariableToken {
        self.push_sym(name, VariableType::Sampler)
    }
    /// Delcare a integral number. Compute pipeline's work group size and
    /// graphics pipeline's vertex count, instance count are of this type. 
    pub fn count(&mut self, name: &str) -> VariableToken {
        self.push_sym(name, VariableType::Count)
    }

    /// Trace the flow head to generate a flow graph.
    pub fn graph(&self, head: &FlowHead) -> FlowGraph {
        FlowGraph::new(head, &self.flows)
    }
}

pub struct TaskInner {
    graph: FlowGraph,
    sym_map: HashMap<String, VariableType>,
}
impl_ptr_wrapper!(Task -> TaskInner);
pub struct Task(Arc<TaskInner>);
impl Task {
    // TODO: Pass in dev to ensure all pipelines are constructed on the same
    // device.
    pub fn new<'a, F>(graph_fn: F) -> Task
        where F: FnOnce(&mut SymbolSource) -> FlowGraph
    {
        let mut sym_src = SymbolSource::new();
        let graph = graph_fn(&mut sym_src);
        let SymbolSource { syms, name_map, .. } = sym_src;
        let sym_map = name_map.into_iter()
            .map(|(name, token)| (name, syms[token]))
            .collect::<HashMap<_, _>>();
        let inner = TaskInner { graph, sym_map };
        Task(Arc::new(inner))
    }
}

/*
pub struct TransactionInner {
    cmdpool: Vec<vk::CommandPool>,
    /// Mapping from queue to command pool.
    queue_map: HashMap,////////////////////////////////////////////////////////
    cmdbufs: Vec<vk::CommandBuffer>,
}
pub struct Transaction(Arc<TransactionInner>);
impl Transaction {
    pub fn new(dev: &Device) -> Result<Transaction> {
        let cmdpool = {
            let create_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index()
                .build();
            let cmdpool = unsafe {
                dev.0.dev.create_command_pool(create_info, None)?
            };
            cmdpool
        }
        let inner = TransactionInner { cmdpool, cmdbufs };
        Transaction(Arc::new(inner))
    }
    pub fn commit(&mut self) {
        
    }
}
*/









