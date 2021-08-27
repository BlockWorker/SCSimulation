#pragma once

#include <iterator>
#include <stdint.h>

namespace scsim {

    class Range
    {
    public:

        const uint32_t from, to;

        class iterator : public std::iterator<std::forward_iterator_tag, uint32_t>
        {
        public:

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
