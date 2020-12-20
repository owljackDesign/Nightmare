#pragma once
#include "GLFW_internal/glfw_base.h"
#if !defined(GLFW_INCLUDE_VULKAN)
    #include <vulkan/vulkan.h>
#endif /* Vulkan header */
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <set>
#include <optional>

#include "vk_mem_alloc/vk_mem_alloc.h"
//This file holds the main vulkan device and memory info.
namespace Nightmare
{
	namespace NightmareRenderer
	{
    #if defined(DEBUG)
        const bool enableValidationLayers = true;
        //const bool enableValidationLayers = false;
    #else
        const bool enableValidationLayers = false;
    #endif

        struct UniformDataContainer
        {
            void* data;
            uint32_t size;
        };

        class VulkanBase
        {
        public:
            VulkanBase(GLFWContainer& glfwContainer) :
                glfwContainer(glfwContainer)
            {
                auto vulkanExtensions = glfwContainer.GetExtensionsForVulkan();
                createInstance(vulkanExtensions.extensionsCount, vulkanExtensions.strings);
                createSurface();
                pickPhysicalDevice();
                createLogicalDevice();
                createSwapChain();
                createSwapChainImageViews();
                initVMA();
                createDepthBuffer();
                for (int i = 0; i < 3; i++)
                {
                    createCommandPool(physicalDevice, commandPools.framePools[syncObjects.currentFrame].cmdPoolHandle, VK_NULL_HANDLE);
                }
                createCommandPool(physicalDevice, commandPools.transientPool, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
            }
            ~VulkanBase()
            {
                for (auto dataContainers : descriptorInfo.dataContainers)
                {
                    free(dataContainers.data);
                }
                descriptorInfo.dataContainers.clear();

                for (auto image : swapChainInfo.imageViews)
                {
                    vkDestroyImageView(device, image, nullptr);
                }
                swapChainInfo.imageViews.clear();

                vkDestroySwapchainKHR(device, swapChainInfo.vulkanHandle, nullptr);
                vkDestroyDevice(device, nullptr);
                vkDestroySurfaceKHR(instance, surface, nullptr);
                vkDestroyInstance(instance, nullptr);
            }

            void CreateDefaultRenderPass()
            {
                VkAttachmentDescription colorAttachment{};
                colorAttachment.format = swapChainInfo.imageFormat;
                colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
                colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

                VkAttachmentDescription depthAttachment{};
                depthAttachment.format = VK_FORMAT_D32_SFLOAT_S8_UINT;
                depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
                depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

                VkAttachmentReference colorAttachmentRef{};
                colorAttachmentRef.attachment = 0;
                colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

                VkAttachmentReference depthAttachmentRef{};
                depthAttachmentRef.attachment = 1;
                depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

                VkSubpassDescription subpass{};
                subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
                subpass.colorAttachmentCount = 1;
                subpass.pColorAttachments = &colorAttachmentRef;
                subpass.pDepthStencilAttachment = &depthAttachmentRef;

                VkSubpassDependency dependency{};
                dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
                dependency.dstSubpass = 0;
                dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                dependency.srcAccessMask = 0;
                dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

                std::array<VkAttachmentDescription, 2> allAttachments{ colorAttachment, depthAttachment };

                VkRenderPassCreateInfo renderPassInfo{};
                renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
                renderPassInfo.attachmentCount = allAttachments.size();
                renderPassInfo.pAttachments = allAttachments.data();
                renderPassInfo.subpassCount = 1;
                renderPassInfo.pSubpasses = &subpass;
                renderPassInfo.dependencyCount = 1;
                renderPassInfo.pDependencies = &dependency;

                if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create render pass!");
                }
            }

            void AddUniform(VkDescriptorType type, uint32_t count, VkShaderStageFlags shaderFlags, UniformDataContainer& dataContainer)
            {
                VkDescriptorSetLayoutBinding uboLayoutBinding{};
                uboLayoutBinding.binding = descriptorInfo.descriptorLayouts.size();
                uboLayoutBinding.descriptorType = type;
                uboLayoutBinding.descriptorCount = count;
                uboLayoutBinding.stageFlags = shaderFlags;
                descriptorInfo.descriptorLayouts.push_back(uboLayoutBinding);
                UniformDataContainer localContainer{};
                if (dataContainer.data != nullptr)
                {
                    localContainer.data = malloc(dataContainer.size);
                    memcpy(localContainer.data, dataContainer.data, dataContainer.size);
                    localContainer.size = dataContainer.size;
                }
                descriptorInfo.dataContainers.push_back(localContainer);

                //UniformBufferAllocation.resize(swapChainImages.size());
                //UniformBufferTriangleAllocation.resize(swapChainImages.size());
                //uniformBuffers.resize(swapChainImages.size());
                //uniformBuffersTriangles.resize(swapChainImages.size());
                //VkDeviceSize bufferSize = sizeof(UniformBufferObject);
                //VkBufferCreateInfo bufferInfo{};
                //bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                //bufferInfo.size = bufferSize;
                //bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                //bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

                //VmaAllocationCreateInfo allocationInfo{};
                //allocationInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
                //for (size_t i = 0; i < swapChainImages.size(); i++)
                //{
                //    vmaCreateBuffer(vmaAllocator, &bufferInfo, &allocationInfo, &uniformBuffers[i], &UniformBufferAllocation[i], nullptr);
                //}
            }

            void UpdateUniform()
            {

            }

            void SelectShader(std::string& vert, std::string& frag)
            {
                shaderInfo.vertFileLocation = vert;
                shaderInfo.fragFileLocation = frag;
                auto vertShaderCode = readFile(shaderInfo.vertFileLocation);
                auto fragShaderCode = readFile(shaderInfo.fragFileLocation);
                shaderInfo.vertShaderModule = createShaderModule(vertShaderCode);
                shaderInfo.fragShaderModule = createShaderModule(fragShaderCode);
            }

            void drawFrame()
            {
                //Wait for current frame to be done rendering.
                std::vector<VkSemaphore> allSemaphores;
                uint32_t imageIndex;
                VkResult ret_val = vkAcquireNextImageKHR(device, swapChainInfo.vulkanHandle, UINT64_MAX, syncObjects.imageAvailableSemaphore[syncObjects.currentFrame], VK_NULL_HANDLE, &imageIndex);
                if (ret_val != VK_SUCCESS)
                {
                    throw std::runtime_error("Failed to aquire next image from swapchain!");
                }
                // Check if a previous frame is using this image (i.e. there is its fence to wait on)
                if(syncObjects.imagesInFlight[imageIndex] != VK_NULL_HANDLE)
                {
                    vkWaitForFences(device, 1, &syncObjects.inFlightFences[syncObjects.currentFrame], VK_TRUE, UINT64_MAX);
                }
                syncObjects.imagesInFlight[imageIndex] = syncObjects.inFlightFences[syncObjects.currentFrame];

                //Reset data
                vkResetCommandPool(device, commandPools.framePools[syncObjects.currentFrame].cmdPoolHandle, 0);
                vkResetCommandPool(device, commandPools.transientPool, 0);
                //update uniform(?)
                //Fully Set uniforms
                //set up pipeline
                //

                CreateDefaultRenderPass();
                createFramebuffers();
                setupUniforms();
                setupPipeline();

                VkSubmitInfo submitInfo{};
                submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                VkSemaphore waitSemaphores[] = { syncObjects.imageAvailableSemaphore[syncObjects.currentFrame] };
                VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
                submitInfo.waitSemaphoreCount = 1;
                submitInfo.pWaitSemaphores = waitSemaphores;
                submitInfo.pWaitDstStageMask = waitStages;
                submitInfo.commandBufferCount = 1;
                submitInfo.pCommandBuffers = &commandPools.framePools[syncObjects.currentFrame].commandBufferHandle;
                VkSemaphore signalSemaphores[] = { syncObjects.renderFinishedSemaphore[currentFrame] };
                allSemaphores.push_back(syncObjects.renderFinishedSemaphore[currentFrame]);
                submitInfo.signalSemaphoreCount = 1;
                submitInfo.pSignalSemaphores = signalSemaphores;
                vkResetFences(device, 1, &syncObjects.inFlightFences[currentFrame]);
                if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
                {
                    throw std::runtime_error("failed to submit draw command buffer!");
                }

            }

        private:
            struct QueueFamilyIndices
            {
                std::optional<uint32_t> graphicsFamily;
                std::optional<uint32_t> presentFamily;
                bool isComplete()
                {
                    return (graphicsFamily.has_value() && presentFamily.has_value());
                }
            };
            struct VulkanQueues {
                VkQueue PresentQueue;
                VkQueue GraphicsQueue;
            };
            struct SwapChainSupportDetails
            {
                VkSurfaceCapabilitiesKHR capabilities;
                std::vector<VkSurfaceFormatKHR> formats;
                std::vector<VkPresentModeKHR> presentModes;
            };
            struct VMAContainer {
                VmaAllocator allocator;
            };
            struct SwapChainInfo
            {
                VkSwapchainKHR vulkanHandle;
                std::vector<VkImage> images;
                std::vector<VkImageView> imageViews;
                VkFormat imageFormat;
                VkExtent2D extent;

                std::vector<VkImage> depthImages;
                std::vector<VkImageView> depthImageViews;
                std::vector<VmaAllocation> depthImageAllocations;

                std::vector<VkFramebuffer> framebuffers;
            };

            struct VulkanDescriptorInfo
            {
                std::vector<VkDescriptorSetLayoutBinding> descriptorLayouts;
                VkDescriptorSetLayout VulkanDescriptorLayoutHandle;
                std::vector <UniformDataContainer> dataContainers;
            };

            struct ShaderInfo
            {
                std::string vertFileLocation;
                VkShaderModule vertShaderModule;
                std::string fragFileLocation;
                VkShaderModule fragShaderModule;
            };

            struct FramePoolInfo
            {
                VkCommandPool cmdPoolHandle;
                VkCommandBuffer commandBufferHandle;
            };

            struct CommandPools
            {
                FramePoolInfo framePools[3];
                VkCommandPool transientPool;
            };

            struct PipelineInfo
            {
                VkPipelineLayout pipelinelayout;
                VkPipeline graphicsPipeline;
            };

            //glfw
            GLFWContainer& glfwContainer;
            VkSurfaceKHR surface;
            VkPhysicalDevice physicalDevice;
            VkInstance instance;
            VkDevice device;
            VulkanQueues queues;
            VMAContainer vmaContainer;
            SwapChainInfo swapChainInfo;
            VkRenderPass renderPass;
            VulkanDescriptorInfo descriptorInfo;
            ShaderInfo shaderInfo;
            CommandPools commandPools;
            PipelineInfo pipelineInfo;

            struct SyncObjects
            {
                VkFence inFlightFences[3];
                VkFence imagesInFlight[3];
                //VkSemaphore imageAvailableSemaphore[3];
                //VkSemaphore renderFinishedSemaphore[3];
                uint32_t currentFrame;
            } syncObjects;

            const std::vector<const char*> validationLayers =
            {
                "VK_LAYER_KHRONOS_validation"
            };
            const std::vector<const char*> deviceExtensions =
            {
                VK_KHR_SWAPCHAIN_EXTENSION_NAME
            };

            void createInstance(uint32_t glfwExtensionCount, const char** glfwExtensions)
            {
                VkApplicationInfo appInfo{};
                appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
                appInfo.pApplicationName = "Nightmare";
                appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
                appInfo.pEngineName = "No Engine";
                appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
                appInfo.apiVersion = VK_API_VERSION_1_0;

                VkInstanceCreateInfo createInfo{};
                createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
                createInfo.pApplicationInfo = &appInfo;

                createInfo.enabledExtensionCount = glfwExtensionCount;
                createInfo.ppEnabledExtensionNames = glfwExtensions;

                /*Get needed Vulkan extensions for GLFW*/
                uint32_t extensionCount;
                vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
                std::vector<VkExtensionProperties> extensions(extensionCount);
                vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

                /*Confirm that Needed extensions are supported by Vulkan*/
                std::cout << "GLFW needed Vulkan extensions: \n";
                for (uint32_t currCount = 0; *glfwExtensions != NULL && currCount < glfwExtensionCount; glfwExtensions++, currCount++)
                {
                    const char* extensionString = *glfwExtensions;
                    bool found = false;
                    std::cout << '\t' << extensionString << " : ";
                    for (const auto& extension : extensions)
                    {
                        if (strcmp(extension.extensionName, extensionString))
                        {
                            found = true;
                            std::cout << " FOUND\n";
                            break;
                        }
                    }
                    if (!found)
                    {
                        std::cout << " NOT FOUND";
                        throw std::runtime_error("failed to find needed extension");
                    }
                }
                std::cout << " ALL EXTENSIONS FOUND\n";

                if (enableValidationLayers && !checkValidationSupport())
                {
                    throw std::runtime_error("failed to get requested validation layers");
                }
                if (enableValidationLayers)
                {
                    std::cout << "\nValidation Layers have been enabled\n";
                    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                    createInfo.ppEnabledLayerNames = validationLayers.data();
                }
                else
                {
                    createInfo.enabledLayerCount = 0;
                }

                if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
                {
                    throw std::runtime_error("failed to create instance");
                }
            }

            /* Confirm that validation layers are supported*/
            bool checkValidationSupport()
            {
                uint32_t layerCount;
                vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
                std::vector<VkLayerProperties> availableLayers(layerCount);
                vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

                for (const char* currentLayerToCheck : validationLayers)
                {
                    bool found = false;
                    std::cout << '\t' << currentLayerToCheck << " : ";
                    for (const auto& currentAvailableLayer : availableLayers)
                    {
                        if (strcmp(currentAvailableLayer.layerName, currentLayerToCheck) == 0)
                        {
                            found = true;
                            std::cout << " FOUND\n";
                            break;
                        }
                    }
                    if (!found)
                    {
                        std::cout << " NOT FOUND";
                        return false;
                    }
                }
                return true;
            }

            void createSurface()
            {
                glfwContainer.createSurface(instance, surface);
            }

            void pickPhysicalDevice()
            {
                uint32_t deviceCount = 0;
                vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
                if (deviceCount == 0)
                {
                    throw std::runtime_error("failed to find GPU with Vulkan Support");
                }
                std::vector<VkPhysicalDevice> devices(deviceCount);
                vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
                for (const auto& device : devices)
                {
                    if (isDeviceSuitable(device))
                    {
                        physicalDevice = device;
                        break;
                    }
                }

                if (physicalDevice == VK_NULL_HANDLE)
                {
                    throw std::runtime_error("Failed to get physical Device.");
                }
            }

            bool isDeviceSuitable(const VkPhysicalDevice& device)
            {
                bool extensionsSupported = checkDeviceExtensionSupport(device);
                QueueFamilyIndices indicies = findQueueFamilies(device);
                bool swapChainAdq = false;
                if (extensionsSupported)
                {
                    SwapChainSupportDetails details = querySwapChainSupport(device);
                    swapChainAdq = !details.formats.empty() && !details.presentModes.empty();
                }
                VkPhysicalDeviceFeatures supportedFeatures;
                vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
                return indicies.isComplete() && extensionsSupported && swapChainAdq && supportedFeatures.samplerAnisotropy;
            }

            bool checkDeviceExtensionSupport(VkPhysicalDevice device)
            {
                uint32_t extensionCount;
                vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
                std::vector<VkExtensionProperties> availableExtensions(extensionCount);
                vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
                std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
                for (const auto& extension : availableExtensions)
                {
                    requiredExtensions.erase(extension.extensionName);
                }
                return requiredExtensions.empty();
            }

            QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
            {
                QueueFamilyIndices indices;

                uint32_t queueFamilyCount = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
                std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
                vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
                int i = 0;

                for (const auto& queueFamily : queueFamilies)
                {
                    VkBool32 presentSupport = false;
                    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
                    if (presentSupport)
                    {
                        indices.presentFamily = i;
                    }
                    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                    {
                        indices.graphicsFamily = i;
                    }
                    if (indices.isComplete())
                    {
                        break;
                    }
                    i++;
                }
                return indices;
            }

            VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
            {
                for (const auto& availableFormat : availableFormats)
                {
                    if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                    {
                        return availableFormat;
                    }
                }

                return availableFormats[0];
            }

            VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
            {
                return VK_PRESENT_MODE_FIFO_KHR;
            }

            void createLogicalDevice()
            {
                QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
                std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
                std::set<uint32_t> uniqueuQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

                float queuePriority = 1.0f;
                for (auto const& queueFamily : uniqueuQueueFamilies)
                {
                    VkDeviceQueueCreateInfo queueCreateInfo{};
                    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                    queueCreateInfo.queueFamilyIndex = queueFamily;
                    queueCreateInfo.queueCount = 1;
                    queueCreateInfo.pQueuePriorities = &queuePriority;
                    queueCreateInfos.push_back(queueCreateInfo);
                }
                VkPhysicalDeviceFeatures deviceFeatures{};
                deviceFeatures.samplerAnisotropy = VK_TRUE;
                VkDeviceCreateInfo createInfo{};
                createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
                createInfo.pQueueCreateInfos = queueCreateInfos.data();
                createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
                createInfo.pEnabledFeatures = &deviceFeatures;
                createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
                createInfo.ppEnabledExtensionNames = deviceExtensions.data();

                if (enableValidationLayers)
                {
                    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                    createInfo.ppEnabledLayerNames = validationLayers.data();
                }
                else
                {
                    createInfo.enabledLayerCount = 0;
                }

                if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create logical device!");
                }

                vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &queues.PresentQueue);
                vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &queues.GraphicsQueue);
            }

            void initVMA()
            {
                VmaAllocatorCreateInfo allocatorInfo{};
                allocatorInfo.device = device;
                allocatorInfo.physicalDevice = physicalDevice;
                allocatorInfo.instance = instance;

                vmaCreateAllocator(&allocatorInfo, &vmaContainer.allocator);
            }
            void createSwapChain()
            {
                SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

                VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
                VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
                VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

                uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

                if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
                {
                    imageCount = swapChainSupport.capabilities.maxImageCount;
                }

                VkSwapchainCreateInfoKHR createInfo{};
                createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
                createInfo.surface = surface;
                createInfo.minImageCount = imageCount;
                createInfo.imageFormat = surfaceFormat.format;
                createInfo.imageColorSpace = surfaceFormat.colorSpace;
                createInfo.imageExtent = extent;
                createInfo.imageArrayLayers = 1;
                createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
                uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

                if (indices.graphicsFamily != indices.presentFamily)
                {
                    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
                    createInfo.queueFamilyIndexCount = 2;
                    createInfo.pQueueFamilyIndices = queueFamilyIndices;
                }
                else
                {
                    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
                    createInfo.queueFamilyIndexCount = 0;
                    createInfo.pQueueFamilyIndices = nullptr;
                }
                createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
                createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
                createInfo.presentMode = presentMode;
                createInfo.clipped = VK_TRUE;
                createInfo.oldSwapchain = VK_NULL_HANDLE;
                if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChainInfo.vulkanHandle) != VK_SUCCESS)
                {
                    throw std::runtime_error("failed to create swap chain!");
                }

                vkGetSwapchainImagesKHR(device, swapChainInfo.vulkanHandle, &imageCount, nullptr);
                swapChainInfo.images.resize(imageCount);
                vkGetSwapchainImagesKHR(device, swapChainInfo.vulkanHandle, &imageCount, swapChainInfo.images.data());
                swapChainInfo.imageFormat = surfaceFormat.format;
                swapChainInfo.extent = extent;
            }
            SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
            {
                SwapChainSupportDetails details;
                vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
                uint32_t format_count;
                vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
                if (format_count != 0)
                {
                    details.formats.resize(format_count);
                    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
                }
                uint32_t presentation_count;
                vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentation_count, nullptr);
                if (format_count != 0)
                {
                    details.presentModes.resize(presentation_count);
                    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentation_count, details.presentModes.data());
                }
                return details;
            }

            VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
            {
                if (capabilities.currentExtent.width != UINT32_MAX)
                {
                    return capabilities.currentExtent;
                }
                else
                {
                    int width, height;
                    GLFWExtent extent = glfwContainer.GetFramebufferSize();
                    VkExtent2D actualExtent = { static_cast<uint32_t>(extent.width), static_cast<uint32_t>(extent.height) };

                    actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
                    actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));
                    return actualExtent;
                }
            }
            void createSwapChainImageViews()
            {
                swapChainInfo.imageViews.resize(swapChainInfo.images.size());
                for (size_t i = 0; i < swapChainInfo.images.size(); i++)
                {
                    swapChainInfo.imageViews[i] = createImageView(swapChainInfo.images[i], swapChainInfo.imageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
                }
            }
            VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
            {
                VkImageViewCreateInfo viewInfo{};
                viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                viewInfo.image = image;
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = format;
                viewInfo.subresourceRange.aspectMask = aspectFlags;
                viewInfo.subresourceRange.baseMipLevel = 0;
                viewInfo.subresourceRange.levelCount = 1;
                viewInfo.subresourceRange.baseArrayLayer = 0;
                viewInfo.subresourceRange.layerCount = 1;

                VkImageView imageView;
                if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
                {
                    throw std::runtime_error("failed to create texture image view!");
                }

                return imageView;
            }

            VkShaderModule createShaderModule(const std::vector<char>& code)
            {
                VkShaderModuleCreateInfo createInfo{};
                createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                createInfo.codeSize = code.size();
                createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
                VkShaderModule shaderModule;
                if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create shader module!");
                }
                return shaderModule;
            }

            void setupUniforms()
            {

            }

            void setupPipeline()
            {
                VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
                vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
                vertShaderStageInfo.module = shaderInfo.vertShaderModule;
                vertShaderStageInfo.pName = "main";

                VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
                fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
                fragShaderStageInfo.module = shaderInfo.fragShaderModule;
                fragShaderStageInfo.pName = "main";
                VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

                //auto bindingDescriptions = Vertex::getBindingDescriptions();
                //auto attributeDescriptions = Vertex::getAttributeDescriptions();
                VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
                vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
                vertexInputInfo.vertexBindingDescriptionCount = 1;
                vertexInputInfo.pVertexBindingDescriptions = nullptr;//&bindingDescriptions;
                vertexInputInfo.vertexAttributeDescriptionCount = 1;//static_cast<uint32_t>(attributeDescriptions.size());
                vertexInputInfo.pVertexAttributeDescriptions = nullptr;// attributeDescriptions.data();

                VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
                inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
                inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
                inputAssembly.primitiveRestartEnable = VK_FALSE;

                VkViewport viewport{};
                viewport.x = 0.0f;
                viewport.y = 0.0f;
                viewport.width = (float)swapChainInfo.extent.width;
                viewport.height = (float)swapChainInfo.extent.height;
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;

                VkRect2D scissor{};
                scissor.offset = { 0, 0 };
                scissor.extent = swapChainInfo.extent;

                VkPipelineViewportStateCreateInfo viewportState{};
                viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
                viewportState.viewportCount = 1;
                viewportState.pViewports = &viewport;
                viewportState.scissorCount = 1;
                viewportState.pScissors = &scissor;

                VkPipelineRasterizationStateCreateInfo rasterizer{};
                rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
                rasterizer.depthClampEnable = VK_FALSE;
                rasterizer.rasterizerDiscardEnable = VK_FALSE;
                rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
                rasterizer.lineWidth = 1.0f;
                rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
                rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                rasterizer.depthBiasEnable = VK_FALSE;

                VkPipelineMultisampleStateCreateInfo multisampling{};
                multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
                multisampling.sampleShadingEnable = VK_FALSE;
                multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

                VkPipelineColorBlendAttachmentState colorBlendAttachment{};
                colorBlendAttachment.colorWriteMask =
                    VK_COLOR_COMPONENT_R_BIT
                    | VK_COLOR_COMPONENT_G_BIT
                    | VK_COLOR_COMPONENT_B_BIT
                    | VK_COLOR_COMPONENT_A_BIT;
                colorBlendAttachment.blendEnable = VK_FALSE;

                VkPipelineColorBlendStateCreateInfo colorBlending{};
                colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
                colorBlending.logicOpEnable = VK_FALSE;
                colorBlending.attachmentCount = 1;
                colorBlending.pAttachments = &colorBlendAttachment;

                VkPipelineDepthStencilStateCreateInfo depthStencil{};
                depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
                depthStencil.depthTestEnable = VK_TRUE;
                depthStencil.depthWriteEnable = VK_TRUE;
                depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
                depthStencil.depthBoundsTestEnable = VK_FALSE;
                depthStencil.stencilTestEnable = VK_FALSE;

                VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
                pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
                pipelineLayoutInfo.setLayoutCount = 1;
                
                pipelineLayoutInfo.pSetLayouts = &descriptorInfo.VulkanDescriptorLayoutHandle;
                if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineInfo.pipelinelayout) != VK_SUCCESS)
                {
                    throw std::runtime_error("failed to create pipeline layout!");
                }

                VkGraphicsPipelineCreateInfo createPipelineInfo{};
                createPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
                createPipelineInfo.stageCount = 2;
                createPipelineInfo.pStages = shaderStages;
                createPipelineInfo.pVertexInputState = &vertexInputInfo;
                createPipelineInfo.pInputAssemblyState = &inputAssembly;
                createPipelineInfo.pViewportState = &viewportState;
                createPipelineInfo.pRasterizationState = &rasterizer;
                createPipelineInfo.pMultisampleState = &multisampling;
                createPipelineInfo.pColorBlendState = &colorBlending;
                createPipelineInfo.layout = pipelineInfo.pipelinelayout;
                createPipelineInfo.renderPass = renderPass;
                createPipelineInfo.pDepthStencilState = &depthStencil;
                createPipelineInfo.subpass = 0;

                if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &createPipelineInfo, nullptr, &pipelineInfo.graphicsPipeline) != VK_SUCCESS)
                {
                    throw std::runtime_error("failed to create graphics pipeline!");
                }
            }

            void createDepthBuffer()
            {

                VkFormat format = VK_FORMAT_D32_SFLOAT_S8_UINT;
                VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
                swapChainInfo.depthImages.resize(swapChainInfo.images.size());
                swapChainInfo.depthImageViews.resize(swapChainInfo.imageViews.size());
                swapChainInfo.depthImageAllocations.resize(swapChainInfo.depthImages.size());
                for (uint32_t i = 0; i < swapChainInfo.depthImageViews.size(); i++)
                {
                    createImage(vmaContainer.allocator, swapChainInfo.extent.width, swapChainInfo.extent.height, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, swapChainInfo.depthImages[i], swapChainInfo.depthImageAllocations[i]);
                    swapChainInfo.depthImageViews[i] = createImageView(swapChainInfo.depthImages[i], format, aspectFlags);
                }
            }

            void createFramebuffers()
            {
                swapChainInfo.framebuffers.resize(swapChainInfo.imageViews.size());
                for (size_t i = 0; i < swapChainInfo.framebuffers.size(); i++)
                {
                    std::array<VkImageView, 2> attachments =
                    {

                        swapChainInfo.imageViews[i],
                        swapChainInfo.depthImageViews[i]
                    };

                    VkFramebufferCreateInfo framebufferInfo{};
                    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                    framebufferInfo.renderPass = renderPass;
                    framebufferInfo.attachmentCount = attachments.size();
                    framebufferInfo.pAttachments = attachments.data();
                    framebufferInfo.width = swapChainInfo.extent.width;
                    framebufferInfo.height = swapChainInfo.extent.height;
                    framebufferInfo.layers = 1;

                    if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainInfo.framebuffers[i]) != VK_SUCCESS)
                    {
                        throw std::runtime_error("failed to create framebuffer!");
                    }
                }
            }

            static std::vector<char> readFile(const std::string& filename)
            {
                std::ifstream file(filename, std::ios::ate | std::ios::binary);

                if (!file.is_open()) {
                    throw std::runtime_error("failed to open file!");
                }
                size_t fileSize = (size_t)file.tellg();
                std::vector<char> buffer(fileSize);
                file.seekg(0);
                file.read(buffer.data(), fileSize);
                file.close();

                return buffer;
            }

            static VkImageView createImageView(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
            {
                VkImageViewCreateInfo viewInfo{};
                viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                viewInfo.image = image;
                viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                viewInfo.format = format;
                viewInfo.subresourceRange.aspectMask = aspectFlags;
                viewInfo.subresourceRange.baseMipLevel = 0;
                viewInfo.subresourceRange.levelCount = 1;
                viewInfo.subresourceRange.baseArrayLayer = 0;
                viewInfo.subresourceRange.layerCount = 1;

                VkImageView imageView;
                if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
                {
                    throw std::runtime_error("failed to create texture image view!");
                }
                return imageView;
            }

            static void createImage(VmaAllocator allocator, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkImage& image, VmaAllocation& imageAllocation)
            {
                VkImageCreateInfo imageInfo{};
                imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                imageInfo.imageType = VK_IMAGE_TYPE_2D;
                imageInfo.extent.width = width;
                imageInfo.extent.height = height;
                imageInfo.extent.depth = 1;
                imageInfo.mipLevels = 1;
                imageInfo.arrayLayers = 1;
                imageInfo.format = format;
                imageInfo.tiling = tiling;
                imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                imageInfo.usage = usage;
                imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
                imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                VmaAllocationCreateInfo allocationCreateInfo{};
                allocationCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

                vmaCreateImage(allocator, &imageInfo, &allocationCreateInfo, &image, &imageAllocation, nullptr);
            }
            void createCommandPool(VkPhysicalDevice physicalDevice, VkCommandPool& localCommandPool, VkQueryPoolCreateFlags flags)
            {
                QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
                VkCommandPoolCreateInfo poolInfo{};
                poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
                poolInfo.flags = flags;
                if (vkCreateCommandPool(device, &poolInfo, nullptr, &localCommandPool) != VK_SUCCESS)
                {
                    throw std::runtime_error("failed to create command pool!");
                }
            }

        };
	}
}