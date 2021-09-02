#pragma once

#include <iterator>
#include <stdint.h>

namespace scsim {

    //input iterator implementing an integer range
    class Range
    {
    public:

        const uint32_t from, to;

        class iterator
        {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = uint32_t;
            using difference_type = ptrdiff_t;
            using pointer = uint32_t*;
            using reference = uint32_t&;

            explicit iterator(uint32_t _from, uint32_t _to, uint32_t _num = 0) : from(_from), to(_to), num(_num) {}
            iterator& operator++() { num = to >= from ? num + 1 : num - 1; return *this; }
            iterator operator++(int) { iterator retval = *this; ++(*this); return retval; }
            bool operator==(iterator other) const { return num == other.num; }
            bool operator!=(iterator other) const { return !(*this == other); }
            reference operator*() const { return (uint32_t)num; }

        private:

            const uint32_t from, to;
            uint32_t num = from;

        };

        Range(uint32_t _from, uint32_t _to) : from(_from), to(_to) { }

        iterator start() {
            return iterator(from, to, from);
        }

        iterator end() {
            return iterator(from, to, to >= from ? to + 1 : to - 1);
        }

    };

}
