#pragma once

#include "mutation.h"

class AddMutation : public Mutation
{
public:

    void apply(Individual& individual) override;
};