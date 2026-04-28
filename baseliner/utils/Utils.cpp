#include <baseliner/utils/Utils.hpp>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
namespace Baseliner::Utils {

  auto gen_uuid() -> std::string {
    static const char charset[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    static constexpr uint64_t EPOCH_2010 = 1262304000000ULL;

    thread_local std::mt19937 rng(std::random_device{}());
    thread_local std::uniform_int_distribution<uint32_t> dist(0, 1679615);

    auto now = std::chrono::system_clock::now();
    long ms_since_unix = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    uint64_t time_val = (ms_since_unix > EPOCH_2010) ? (ms_since_unix - EPOCH_2010) : 0;
    uint32_t rand_val = dist(rng);

    std::string out(13, '0');

    for (int i = 8; i >= 0; --i) {
      out[i] = charset[time_val % 36];
      time_val /= 36;
    }

    for (int i = 12; i >= 9; --i) {
      out[i] = charset[rand_val % 36];
      rand_val /= 36;
    }
    return out;
  }

  auto get_datetime() -> std::string {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = std::chrono::system_clock::to_time_t(now);

    std::tm bt = *std::localtime(&timer);
    std::ostringstream oss;

    oss << std::put_time(&bt, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count();

    return oss.str();
  }
} // namespace Baseliner::Utils