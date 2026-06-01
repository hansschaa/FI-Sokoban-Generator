#pragma once

#include "mutation.h"

class RemoveMutation : public Mutation
{
public:

    bool apply(Individual& individual) override;
};