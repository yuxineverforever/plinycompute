#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <functional>
#include <list>
#include <utility>
#include <algorithm>
#include <shared_mutex>

template<typename Key, typename Value>
class threadSafeMap{

public:
    std::map<Key, Value> tsMap;
    mutable std::shared_timed_mutex mutex;

public:

    threadSafeMap() = default;
    /**
     * Insert a key,value pair into Map. Dont allow duplicated keys.
     * @param key
     * @param value
     * @return
     */
    bool insert(Key & key, Value& value){
        std::unique_lock<std::shared_timed_mutex> lock(mutex);
        if (tsMap.count(key) != 0){
            std::cerr << "Insert a duplicated key into thread safe map!\n";
            return false;
        }
        auto res = tsMap.insert(std::make_pair(key, value));
        return res.second;
    }

    /**
     * For simplicity, this function is only used for retrieve value from map.
     * @param key
     * @return
     */
    Value& operator [](const Key& key){
        std::shared_lock<std::shared_timed_mutex> lock(mutex);
        return tsMap[key];
    }

    /**
     * Count if the key is existed in the Map.
     * @param key
     * @return
     */
    uint32_t count(Key& key){
        std::shared_lock<std::shared_timed_mutex> lock(mutex);
        return tsMap.count(key);
    }

    /**
     *
     * @return size of the map
     */
    size_t length(){
        std::shared_lock<std::shared_timed_mutex> lock(mutex);
        return tsMap.size();
    }

};