#include <cassert>
#include <stdint.h>

class HighLowByte
{
public:
    HighLowByte() = default;

    constexpr HighLowByte(const uint8_t low_nibble, const uint8_t high_nibble)
    {
        set_high_nibble(high_nibble);
        set_low_nibble(low_nibble);
    }

    constexpr uint8_t get_low_nibble() const 
    {
        return data & 0b0001111;
    }

    constexpr uint8_t get_high_nibble() const 
    {
        return data >> 4;
    }

    void set_low_nibble(const uint8_t value)  
    {
        assert(value <= 0b0001111);
        data = (data & 0b11110000) | value;
    }

    void set_high_nibble(const uint8_t value) 
    {
        assert(value <= 0b0001111);
        data = (data & 0b00001111) | (value << 4);
    }

    constexpr uint8_t get_data() const 
    {
        return data;
    }
private:
    uint8_t data = 0;
};