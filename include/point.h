#pragma once
#include <cstdint>

class point {

public:
    int16_t x;
    int16_t y;
    point(int16_t xx = 0, int16_t yy = 0);
    point(const point &a);
    point operator+(const point &a)const;
    point operator-(const point &a)const;
    point operator*(int16_t m) const;
    point& operator=(const point &a);
    bool operator==(const point &a)const;
    bool operator!=(const point &a)const;
    bool operator<(const point &a)const;
    void show()const;
};