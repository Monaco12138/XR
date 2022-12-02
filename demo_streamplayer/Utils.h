//
// Created by ubuntu on 4/1/22.
//

#ifndef _UTILS_
#define _UTILS_

#include <chrono>

using TimePoint = std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>;
using Clock = std::chrono::high_resolution_clock;

class Utils {
public:
    
    template<class T>
    static bool is_none_empty(std::vector<std::shared_ptr<T>> queue_list) {
        return !std::any_of(queue_list.begin(), queue_list.end(), [&](const std::shared_ptr<T> &x) {
            return x->empty();
        });
    }

    template<class T>
    static bool is_all_empty(std::vector<std::shared_ptr<T>> queue_list) {
        return std::all_of(queue_list.begin(), queue_list.end(), [&](const std::shared_ptr<T> &x) {
            return x->empty();
        });
    }
    
    static TimePoint clock() {
        return Clock::now();
    }

    template<typename T = std::chrono::milliseconds>
    static int get_duration(TimePoint start, TimePoint end) {
        return int(std::chrono::duration_cast<T>(end - start).count());
    }

    template<typename T = std::chrono::milliseconds>
    static int get_time_point(TimePoint time) {

        return int(std::chrono::duration_cast<T>( time.time_since_epoch() ).count());
    }
};

#endif //_UTILS_
