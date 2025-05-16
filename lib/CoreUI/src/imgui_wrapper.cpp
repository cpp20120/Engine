#include "../include/imgui_wrapper.hpp"
#include <stdio.h>

int main() {
  if (!glfwInit()) return -1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Create window
  GLFWwindow* window =
      glfwCreateWindow(800, 600, "ImGui Button Example", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  // Initialize GLAD
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    glfwTerminate();
    return -1;
  }

  // Setup ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;

  // Setup ImGui style
  ImGui::StyleColorsDark();


  // Main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();


    ImGui::NewFrame();

    // Create a simple window with a button
    ImGui::Begin("Demo Window");
    if (ImGui::Button("Click Me!")) {
      // Button was clicked - do something
      printf("Button clicked!\n");
    }
    ImGui::End();

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(window);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}