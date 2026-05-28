#pragma once

#include "mutation.h"

class MoveMutation : public Mutation {

public:

    void apply(Individual& ind) override;
};