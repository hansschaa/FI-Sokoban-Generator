#include <cstdio>
#include "point.h"

// FIX: int8_t limita coordenadas a [-128, 127].
// Mapas de más de 127 filas/columnas causaban overflow silencioso.
// Se usa int16_t: soporta hasta 32767, suficiente para cualquier puzzle Sokoban.
point::point(int16_t xx, int16_t yy) {
    x = xx;
    y = yy;
}

point::point(const point &a) {
    x = a.x;
    y = a.y;
}

point point::operator+(const point &a)const {
    point result(a.x + x, a.y + y);
    return result;
}

point point::operator-(const point &a)const{
    point result(x - a.x, y - a.y);
    return result;
}

// FIX: el multiplicador también era int8_t — mismo problema de overflow.
point point::operator*(int16_t m) const{
    point result(x*m, y*m);
    return result;
}

point& point::operator=(const point &a) {
    x = a.x;
    y = a.y;
    return *this;
}

bool point::operator==(const point &a)const {
    return (a.x == x && a.y == y);
}

bool point::operator!=(const point &a)const {
    return (a.x != x || a.y != y);
}

bool point::operator<(const point &a)const {
    if (y != a.y) { return y < a.y; }
    return x < a.x;
}

void point::show()const {
    printf("x:%d y:%d", int(x), int(y));
}