#pragma once

#include <stdexcept>

#include "compact_elias_fano.hpp"
#include "compact_ranked_bitvector.hpp"
#include "all_ones_sequence.hpp"
#include "global_parameters.hpp"

namespace ds2i {

    struct indexed_sequence {

        enum index_type {
            elias_fano = 0,
            ranked_bitvector = 1,
            all_ones = 2,
        };
        using Cost = uint64_t;
        using CompressionCost = std::pair<index_type, Cost>;

        static DS2I_FLATTEN_FUNC CompressionCost best_compressor(
            global_parameters const &params, uint64_t universe, uint64_t n) {
          Cost ef_cost = compact_elias_fano::bitsize(params, universe, n);
          Cost rb_cost = compact_ranked_bitvector::bitsize(params, universe, n);
          Cost aos_cost = all_ones_sequence::bitsize(params, universe, n);
          std::vector<CompressionCost> type_cost = {{elias_fano, ef_cost},
                                                    {ranked_bitvector, rb_cost},
                                                    {all_ones, aos_cost}};
          return *std::min_element(type_cost.begin(), type_cost.end(),
                                   [](const auto &lhs, const auto &rhs) {
                                     return lhs.second < rhs.second;
                                   });
        }

        static DS2I_FLATTEN_FUNC uint64_t
        bitsize(global_parameters const& params, uint64_t universe, uint64_t n)
        {
            return best_compressor(params, universe, n).second;
        }

        template <typename Iterator>
        static void write(succinct::bit_vector_builder& bvb,
                          Iterator begin,
                          uint64_t universe, uint64_t n,
                          global_parameters const& params)
        {
            int best_type = best_compressor(params, universe, n).first;


            switch (best_type) {
            case elias_fano:
                compact_elias_fano::write(bvb, begin,
                                          universe, n,
                                          params);
                break;
            case ranked_bitvector:
                compact_ranked_bitvector::write(bvb, begin,
                                                universe, n,
                                                params);
                break;
            case all_ones:
                all_ones_sequence::write(bvb, begin,
                                         universe, n,
                                         params);
                break;
            default:
                assert(false);
            }
        }

        class enumerator {
        public:

            typedef std::pair<uint64_t, uint64_t> value_type; // (position, value)

            enumerator()
            {}

            enumerator(succinct::bit_vector const& bv, uint64_t offset,
                       uint64_t universe, uint64_t n,
                       global_parameters const& params)
            {
                m_type = best_compressor(params, universe, n).first;
                switch (m_type) {
                case elias_fano:
                    m_ef_enumerator = compact_elias_fano::enumerator(bv, offset,
                                                                     universe, n,
                                                                     params);
                    break;
                case ranked_bitvector:
                    m_rb_enumerator = compact_ranked_bitvector::enumerator(bv, offset,
                                                                           universe, n,
                                                                           params);
                    break;
                case all_ones:
                    m_ao_enumerator = all_ones_sequence::enumerator(bv, offset,
                                                                    universe, n,
                                                                    params);
                    break;
                default:
                    throw std::invalid_argument("Unsupported type");
                }
            }

#define ENUMERATOR_METHOD(RETURN_TYPE, METHOD, FORMALS, ACTUALS)    \
            RETURN_TYPE DS2I_FLATTEN_FUNC METHOD FORMALS              \
            {                                                       \
                switch (__builtin_expect(m_type, elias_fano)) {     \
                case elias_fano:                                    \
                    return m_ef_enumerator.METHOD ACTUALS;          \
                case ranked_bitvector:                              \
                    return m_rb_enumerator.METHOD ACTUALS;          \
                case all_ones:                                      \
                    return m_ao_enumerator.METHOD ACTUALS;          \
                default:                                            \
                    assert(false);                                  \
                    __builtin_unreachable();                        \
                }                                                   \
            }                                                       \
            /**/

            // semicolons are redundant but they are needed to get emacs to
            // align the lines properly
            ENUMERATOR_METHOD(value_type, move, (uint64_t position), (position));
            ENUMERATOR_METHOD(value_type, next_geq, (uint64_t lower_bound), (lower_bound));
            ENUMERATOR_METHOD(value_type, next, (), ());
            ENUMERATOR_METHOD(uint64_t, size, () const, ());
            ENUMERATOR_METHOD(uint64_t, prev_value, () const, ());

#undef ENUMERATOR_METHOD
#undef ENUMERATOR_VOID_METHOD

        private:
            index_type m_type;
            union {
                compact_elias_fano::enumerator m_ef_enumerator;
                compact_ranked_bitvector::enumerator m_rb_enumerator;
                all_ones_sequence::enumerator m_ao_enumerator;
            };
        };
    };
}