#include <atomic>
#include <baseliner/Result.hpp>
#include <iomanip>
#include <random>
#include <sstream>

namespace Baseliner {
  static std::atomic<uint8_t> counter{0};

  std::string Result::current_time_string() {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S");
    return ss.str();
  };
  std::string Result::generate_uid() {
    using namespace std::chrono;
    auto now = duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 0xFFFF);
    uint16_t rand_val = dis(gen);

    uint8_t count = counter.fetch_add(1, std::memory_order_relaxed);
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << now << std::setw(2) << static_cast<int>(count)
       << std::setw(4) << rand_val;
    return ss.str();
  };
} // namespace Baseliner