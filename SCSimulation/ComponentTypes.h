#pragma once

#define COMPONENT_COMBINATORIAL 0x00000000u
#define COMPONENT_SEQUENTIAL 0x80000000u

#define COMPONENT_INV (COMPONENT_COMBINATORIAL | 1u)
#define COMPONENT_AND (COMPONENT_COMBINATORIAL | 2u)
#define COMPONENT_NAND (COMPONENT_COMBINATORIAL | 3u)
#define COMPONENT_OR (COMPONENT_COMBINATORIAL | 4u)
#define COMPONENT_NOR (COMPONENT_COMBINATORIAL | 5u)
#define COMPONENT_XOR (COMPONENT_COMBINATORIAL | 6u)
#define COMPONENT_XNOR (COMPONENT_COMBINATORIAL | 7u)
#define COMPONENT_MUX2 (COMPONENT_COMBINATORIAL | 8u)
#define COMPONENT_MUXN (COMPONENT_COMBINATORIAL | 9u)

#define COMPONENT_STANH (COMPONENT_SEQUENTIAL | 1u)