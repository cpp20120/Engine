
#include "../include/include.hpp"
//#include "../lib/CoreMeta/include/concepts/utility_concepts.hpp"
//#include "../lib/CoreMeta/include/meta_funcs.hpp"

#include "../lib/CoreUI/include/imgui_wrapper.hpp"
#include "../lib/CoreUI/src/imgui_wrapper.cpp"

int main() {
  proj::func(1, 2);

  constexpr int first_number = 23;
  constexpr int second_number = 45;
  window();  // core::meta::concepts::printable<T>;
  return 0;
}