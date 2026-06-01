#pragma once

#include "mutation.h"

class AddMutation : public Mutation
{
public:

    bool apply(Individual& individual) override;
    
};