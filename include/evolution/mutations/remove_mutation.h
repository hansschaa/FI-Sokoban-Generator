#pragma once

#include "mutation.h"

class RemoveMutation : public Mutation
{
public:

    void apply(Individual& individual) override;
};