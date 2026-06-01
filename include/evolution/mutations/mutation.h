#pragma once

#include "../individual.h"

class Mutation {

public:

    virtual bool apply(Individual&) = 0;
};