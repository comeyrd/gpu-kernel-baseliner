#ifndef NOTHING_KERNEL_HPP
#define NOTHING_KERNEL_HPP
#include <baseliner/Case.hpp>

namespace Baseliner {
  template <typename BackendT>
  class NothingKernel : public ICase<BackendT> {
  public:
    using backend = BackendT;
    auto name() -> std::string override {
      return "NothingKernel";
    };
    auto validate_case() -> bool override {
      return true;
    }
    void setup(std::shared_ptr<typename backend::stream_t> stream) override;
    void reset_case(std::shared_ptr<typename backend::stream_t> stream) override;
    void run_case(std::shared_ptr<typename backend::stream_t> stream) override;
    void teardown(std::shared_ptr<typename backend::stream_t> stream) override;
    void register_options() override {
      ICase<BackendT>::register_options();
      this->add_option("NothingKernel", "async_memcpy", "Should the copy before the launch be async?", m_async_memcpy);
      this->add_option("NothingKernel", "nb_bytes",
                       "How many bytes should be copied | memset before launching the empty kernel ?", m_bytes_copied);
      this->add_option("NothingKernel", "blocks", "Numbers of block to be launched in", m_blocks);
      this->add_option("NothingKernel", "threads", "Numbers of threads to be launched in", m_threads);
    };

  private:
    int m_blocks = 256;
    int m_threads = 256;
    char *m_d_buffer;
    bool m_async_memcpy = false;
    size_t m_bytes_copied = 10;
  };

} // namespace Baseliner
#endif // NOTHING_KERNEL_HPP