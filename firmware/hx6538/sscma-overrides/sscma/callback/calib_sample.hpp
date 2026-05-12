#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <string>

#include "core/utils/el_base64.h"
#include "sscma/definations.hpp"
#include "sscma/static_resource.hpp"
#include "sscma/utility.hpp"

namespace sscma::callback {

using namespace sscma::utility;

class CalibSample final : public std::enable_shared_from_this<CalibSample> {
   public:
    std::shared_ptr<CalibSample> getptr() { return shared_from_this(); }

    [[nodiscard]] static std::shared_ptr<CalibSample> create(std::string cmd, int32_t n_times, void* caller) {
        return std::shared_ptr<CalibSample>{
          new CalibSample{std::move(cmd), n_times, caller}
        };
    }

    ~CalibSample() { static_resource->is_sample = false; }

    inline void run() { prepare(); }

   protected:
    CalibSample(std::string cmd, int32_t n_times, void* caller)
        : _cmd{cmd},
          _n_times{n_times},
          _caller{caller},
          _task_id{static_resource->current_task_id.load(std::memory_order_seq_cst)},
          _ret{EL_OK} {
        static_resource->is_sample = true;
    }

   private:
    static constexpr std::size_t RAW_CHUNK_SIZE    = 3072;
    static constexpr std::size_t BASE64_CHUNK_SIZE = ((RAW_CHUNK_SIZE + 2u) / 3u) << 2u;

    inline void prepare() {
        prepare_sensor_info();
        if (_n_times != 1) [[unlikely]] {
            _ret = EL_EINVAL;
            direct_reply();
            return;
        }
        if (!check_sensor_status()) [[unlikely]]
            goto Err;

        static_resource->executor->add_task(
          [_this = std::move(getptr())](const std::atomic<bool>&) { _this->event_loop(); });
        return;

    Err:
        direct_reply();
    }

    inline void prepare_sensor_info() {
        _sensor_info = static_resource->device->get_sensor_info(static_resource->current_sensor_id);
    }

    inline bool check_sensor_status() {
        _ret = _sensor_info.id != 0 ? EL_OK : EL_EIO;
        if (_ret != EL_OK) [[unlikely]]
            return false;
        _ret = _sensor_info.state == EL_SENSOR_STA_AVAIL ? EL_OK : EL_EIO;
        if (_ret != EL_OK) [[unlikely]]
            return false;
        return true;
    }

    inline void direct_reply() {
        auto ss{concat_strings("\r{\"type\": 0, \"name\": \"",
                               _cmd,
                               "\", \"code\": ",
                               std::to_string(_ret),
                               ", \"data\": {\"sensor\": ",
                               sensor_info_2_json_str(_sensor_info, static_resource->device),
                               "}}\n")};
        static_cast<Transport*>(_caller)->send_bytes(ss.c_str(), ss.size());
    }

    inline const char* jpeg_qtable() const {
        return _sensor_opt_id == 5 ? "JPEG_ENC_QTABLE_4X" : "module_default";
    }

    inline std::string event_prefix(const char* phase, int32_t chunk_index) const {
        return concat_strings("\r{\"type\": 1, \"name\": \"",
                              _cmd,
                              "\", \"code\": ",
                              std::to_string(_ret),
                              ", \"data\": {\"phase\": \"",
                              phase,
                              "\", \"resolution\": [",
                              std::to_string(_width),
                              ", ",
                              std::to_string(_height),
                              "], \"jpeg_byte_count\": ",
                              std::to_string(_jpeg_byte_count),
                              ", \"base64_length\": ",
                              std::to_string(_base64_length),
                              ", \"chunk_index\": ",
                              std::to_string(chunk_index),
                              ", \"chunk_count\": ",
                              std::to_string(_chunk_count),
                              ", \"sensor_opt_id\": ",
                              std::to_string(_sensor_opt_id),
                              ", \"jpeg_qtable\": \"",
                              jpeg_qtable(),
                              "\"");
    }

    inline void event_reply_no_chunk(const char* phase, int32_t chunk_index) {
        auto ss{concat_strings(event_prefix(phase, chunk_index), "}}\n")};
        static_cast<Transport*>(_caller)->send_bytes(ss.c_str(), ss.size());
    }

    inline void event_reply_chunk(std::size_t chunk_index, const uint8_t* data, std::size_t size) {
        char encoded[BASE64_CHUNK_SIZE + 1]{};
        const std::size_t encoded_size = ((size + 2u) / 3u) << 2u;
        el_base64_encode(data, static_cast<int>(size), encoded);
        encoded[encoded_size] = '\0';

        auto prefix{concat_strings(event_prefix("chunk", static_cast<int32_t>(chunk_index)), ", \"image_chunk\": \"")};
        static_cast<Transport*>(_caller)->send_bytes(prefix.c_str(), prefix.size());
        static_cast<Transport*>(_caller)->send_bytes(encoded, encoded_size);
        static_cast<Transport*>(_caller)->send_bytes("\"}}\n", 4);
    }

    inline void prepare_frame_metadata(const el_img_t& frame) {
        auto camera       = static_resource->device->get_camera();
        _width            = frame.width;
        _height           = frame.height;
        _jpeg_byte_count  = frame.size;
        _base64_length    = ((frame.size + 2u) / 3u) << 2u;
        _chunk_count      = (frame.size + RAW_CHUNK_SIZE - 1u) / RAW_CHUNK_SIZE;
        _sensor_opt_id    = static_cast<int32_t>(camera->current_opt_id());
    }

    inline void event_loop() {
        switch (_sensor_info.type) {
        case EL_SENSOR_TYPE_CAM:
            direct_reply();
            return event_loop_cam();
        default:
            _ret = EL_ENOTSUP;
            direct_reply();
        }
    }

    void event_loop_cam() {
        if (static_resource->current_task_id.load(std::memory_order_seq_cst) != _task_id) [[unlikely]]
            return;

        auto camera       = static_resource->device->get_camera();
        auto frame        = el_img_t{};
        bool stream_open  = false;

        _ret = camera->start_stream();
        if (!is_everything_ok()) [[unlikely]]
            goto Err;
        stream_open = true;

#if CONFIG_EL_HAS_ACCELERATED_JPEG_CODEC
        _ret = camera->get_processed_frame(&frame);
#else
        _ret = EL_ENOTSUP;
#endif
        if (!is_everything_ok() || !frame.data || frame.size == 0) [[unlikely]] {
            if (_ret == EL_OK) _ret = EL_EIO;
            goto Err;
        }

        _ret = camera->stop_stream();
        stream_open = false;
        if (!is_everything_ok()) [[unlikely]]
            goto Err;

        prepare_frame_metadata(frame);
        event_reply_no_chunk("begin", 0);

        for (std::size_t index = 0; index < _chunk_count; ++index) {
            const std::size_t offset     = index * RAW_CHUNK_SIZE;
            const std::size_t chunk_size = std::min<std::size_t>(RAW_CHUNK_SIZE, frame.size - offset);
            event_reply_chunk(index, frame.data + offset, chunk_size);
        }

        event_reply_no_chunk("end", static_cast<int32_t>(_chunk_count));
        return;

    Err:
        if (stream_open) {
            camera->stop_stream();
        }
        event_reply_no_chunk("end", 0);
    }

    inline bool is_everything_ok() const { return _ret == EL_OK; }

   private:
    std::string _cmd;
    int32_t     _n_times;
    void*       _caller;

    std::size_t      _task_id;
    el_sensor_info_t _sensor_info;

    el_err_code_t _ret;
    std::size_t   _width           = 0;
    std::size_t   _height          = 0;
    std::size_t   _jpeg_byte_count = 0;
    std::size_t   _base64_length   = 0;
    std::size_t   _chunk_count     = 0;
    int32_t       _sensor_opt_id   = -1;
};

}  // namespace sscma::callback
