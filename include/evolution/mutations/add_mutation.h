#pragma once

#include "mutation.h"

class AddMutation : public Mutation
{
public:
    std::vector<std::vector<bool>> deadlock_mask;

    bool apply(Individual& individual) override;
    
};