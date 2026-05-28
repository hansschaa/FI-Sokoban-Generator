#pragma once

#include "../individual.h"

class Mutation {

public:

    virtual void apply(Individual& ind) = 0;
};