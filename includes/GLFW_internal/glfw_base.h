#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdexcept>
//This file holds the main vulkan device and memory info.
namespace Nightmare
{

	struct VulkanExtensions
	{
		uint32_t extensionsCount;
		const char** strings;
	};

	struct GLFWExtent
	{
		int width;
		int height;
	};

	class GLFWContainer
	{
	public:
		GLFWContainer(int width, int height)
		{
			glfwInit();
			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
			window = glfwCreateWindow(width, height, "Nightmare", nullptr, nullptr);
		}

		~GLFWContainer()
		{
			glfwDestroyWindow(window);
			glfwTerminate();
		}

		VulkanExtensions GetExtensionsForVulkan()
		{
			VulkanExtensions glfwExtensions;
			glfwExtensions.strings = glfwGetRequiredInstanceExtensions(&glfwExtensions.extensionsCount);
			return glfwExtensions;
		}
		void createSurface(VkInstance instance, VkSurfaceKHR& surface)
		{
			if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) 
			{
				throw std::runtime_error("failed to create window surface!");
			}
		}

		GLFWExtent GetFramebufferSize()
		{
			GLFWExtent extent;
			glfwGetFramebufferSize(window, &extent.width, &extent.height);
			return extent;
		}

	private:
		GLFWwindow* window;
	};
}