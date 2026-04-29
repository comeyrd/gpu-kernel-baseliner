#ifndef NOTHING_WORKLOAD_HPP
#define NOTHING_WORKLOAD_HPP

#include <baseliner/core/Workload.hpp>

#include <string>

namespace Baseliner {

  template <typename BackendT>
  class NothingWorkload : public IWorkload<BackendT> {
  public:
    using backend = BackendT;

    NothingWorkload() = default;

    // Identification
    auto algo() -> std::string override {
      return "Nothing";
    }

    void setup_host_random_generated() override {
    }
    void setup_host_from_file(std::string & /*path*/) override {
    }
    void inner_save_setup(std::string & /*path*/) override {
    }
    void results_from_reference() override {
    }

    void setup_device(std::shared_ptr<typename backend::stream_t> stream) override;
    void reset_device(std::shared_ptr<typename backend::stream_t> stream) override;
    auto run(std::shared_ptr<typename backend::stream_t> stream) -> typename backend::launch_result_t override;
    void fetch_results(std::shared_ptr<typename backend::stream_t> stream) override;

    void free() override {};

    auto validate() -> bool override {
      return true;
    }

  protected:
    void register_options() override {
      IWorkload<BackendT>::register_options();
      this->add_option("NothingWorkload", "async_memcpy", "Should the copy before the launch be async?",
                       m_async_memcpy);
      this->add_option("NothingWorkload", "nb_bytes",
                       "How many bytes should be copied / memset before launching the empty kernel?", m_bytes_copied);
      this->add_option("NothingWorkload", "blocks", "Number of blocks to launch", m_blocks);
      this->add_option("NothingWorkload", "threads", "Number of threads to launch", m_threads);
    }

  private:
    int m_blocks = 256;
    int m_threads = 256;
    bool m_async_memcpy = false;
    size_t m_bytes_copied = 10;

    char *m_d_buffer = nullptr;
  };

} // namespace Baseliner

#endif // NOTHING_WORKLOAD_HPP