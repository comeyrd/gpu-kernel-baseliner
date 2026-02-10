#include <atomic>
#include <baseliner/Result.hpp>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <ios>
#include <random>
#include <sstream>
#include <string>

namespace Baseliner {
  static std::atomic<uint8_t> counter{0}; // NOLINT

  auto Result::current_time_string() -> std::string {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream stringstream;
    stringstream << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S");
    return stringstream.str();
  };
  auto Result::generate_uid() -> std::string {
    using namespace std::chrono;
    auto now = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
    static std::random_device random_g;
    static std::mt19937 gen(random_g());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF); // NOLINT
    const uint16_t rand_val = dis(gen);

    const uint8_t count = counter.fetch_add(1, std::memory_order_relaxed);
    std::stringstream stringstream;
    stringstream << std::hex << std::setfill('0') << std::setw(8)                               // NOLINT
                 << now << std::setw(2) << static_cast<int>(count) << std::setw(4) << rand_val; // NOLINT
    return stringstream.str();
  };
} // namespace Baseliner