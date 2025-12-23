/*
 * poker.h - High-performance Texas Hold'em Limit Poker
 * Optimized for 5M+ steps per second with PufferLib
 */

#ifndef POKER_H
#define POKER_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

// ============================================================================
// Constants
// ============================================================================

#define MAX_PLAYERS 6
#define DECK_SIZE 52
#define HAND_SIZE 2
#define COMMUNITY_SIZE 5
#define NUM_ACTIONS 3  // Fold, Call/Check, Raise

// Card representation: card = suit * 13 + rank
// Suits: 0=Spades, 1=Hearts, 2=Diamonds, 3=Clubs
// Ranks: 0=2, 1=3, ..., 8=10, 9=J, 10=Q, 11=K, 12=A

#define CARD_RANK(c) ((c) % 13)
#define CARD_SUIT(c) ((c) / 13)
#define MAKE_CARD(suit, rank) ((suit) * 13 + (rank))

// Hand rankings (higher = better)
#define HAND_HIGH_CARD     0
#define HAND_PAIR          1
#define HAND_TWO_PAIR      2
#define HAND_THREE_KIND    3
#define HAND_STRAIGHT      4
#define HAND_FLUSH         5
#define HAND_FULL_HOUSE    6
#define HAND_FOUR_KIND     7
#define HAND_STRAIGHT_FLUSH 8

// Observation size per player:
// - 52 bits for hole cards (one-hot)
// - 52 bits for community cards (one-hot)
// - 6 floats for stack sizes (normalized)
// - 1 float for pot size (normalized)
// - 6 floats for player active status
// - 4 floats for betting round (one-hot)
// - 3 floats for last action per player * 6 = 18
// Total: 52 + 52 + 6 + 1 + 6 + 4 + 18 = 139
#define OBS_SIZE 139

// ============================================================================
// Fast PRNG (xorshift64)
// ============================================================================

static inline uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

// ============================================================================
// Hand Evaluation (optimized)
// ============================================================================

typedef struct {
    uint32_t hand_type;   // HAND_HIGH_CARD to HAND_STRAIGHT_FLUSH
    uint32_t hand_value;  // For comparing same type hands
} HandRank;

// Lookup tables for fast evaluation
static uint16_t flush_table[8192];    // 2^13 possible rank combinations
static uint16_t unique5_table[8192];  // Non-flush 5-card hands
static bool tables_initialized = false;

static inline int popcount(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

// Initialize lookup tables
static void init_hand_tables() {
    if (tables_initialized) return;

    // Simple initialization - will be filled with proper values
    for (int i = 0; i < 8192; i++) {
        flush_table[i] = 0;
        unique5_table[i] = 0;
    }

    tables_initialized = true;
}

// Evaluate 7 cards and return best 5-card hand rank
static HandRank evaluate_7cards(uint8_t cards[7]) {
    HandRank result = {HAND_HIGH_CARD, 0};

    // Count ranks and suits
    uint8_t rank_count[13] = {0};
    uint8_t suit_count[4] = {0};
    uint16_t rank_bits = 0;
    uint8_t suit_cards[4][7];
    uint8_t suit_card_count[4] = {0};

    for (int i = 0; i < 7; i++) {
        uint8_t rank = CARD_RANK(cards[i]);
        uint8_t suit = CARD_SUIT(cards[i]);
        rank_count[rank]++;
        suit_count[suit]++;
        rank_bits |= (1 << rank);
        suit_cards[suit][suit_card_count[suit]++] = rank;
    }

    // Check for flush
    int flush_suit = -1;
    for (int s = 0; s < 4; s++) {
        if (suit_count[s] >= 5) {
            flush_suit = s;
            break;
        }
    }

    // Check for straight
    uint16_t straight_mask = rank_bits | (rank_bits >> 13); // Wrap ace
    int straight_high = -1;
    for (int i = 12; i >= 4; i--) {
        uint16_t mask = 0x1F << (i - 4);
        if ((straight_mask & mask) == mask) {
            straight_high = i;
            break;
        }
    }
    // Check wheel (A-2-3-4-5)
    if (straight_high < 0 && (straight_mask & 0x100F) == 0x100F) {
        straight_high = 3; // 5-high straight
    }

    // Check for straight flush
    if (flush_suit >= 0 && straight_high >= 0) {
        uint16_t flush_bits = 0;
        for (int i = 0; i < suit_card_count[flush_suit]; i++) {
            flush_bits |= (1 << suit_cards[flush_suit][i]);
        }
        flush_bits |= (flush_bits >> 13);
        for (int i = 12; i >= 4; i--) {
            uint16_t mask = 0x1F << (i - 4);
            if ((flush_bits & mask) == mask) {
                result.hand_type = HAND_STRAIGHT_FLUSH;
                result.hand_value = i;
                return result;
            }
        }
        if ((flush_bits & 0x100F) == 0x100F) {
            result.hand_type = HAND_STRAIGHT_FLUSH;
            result.hand_value = 3;
            return result;
        }
    }

    // Count pairs, trips, quads
    int pairs = 0, trips = 0, quads = 0;
    int pair_ranks[3] = {-1, -1, -1};
    int trip_rank = -1, quad_rank = -1;

    for (int r = 12; r >= 0; r--) {
        if (rank_count[r] == 4) {
            quads++;
            quad_rank = r;
        } else if (rank_count[r] == 3) {
            trips++;
            trip_rank = r;
        } else if (rank_count[r] == 2) {
            if (pairs < 3) pair_ranks[pairs] = r;
            pairs++;
        }
    }

    // Determine hand type
    if (quads > 0) {
        result.hand_type = HAND_FOUR_KIND;
        result.hand_value = quad_rank * 13;
        // Add kicker
        for (int r = 12; r >= 0; r--) {
            if (rank_count[r] > 0 && r != quad_rank) {
                result.hand_value += r;
                break;
            }
        }
        return result;
    }

    if (trips > 0 && (pairs > 0 || trips > 1)) {
        result.hand_type = HAND_FULL_HOUSE;
        result.hand_value = trip_rank * 13;
        // Find best pair
        for (int r = 12; r >= 0; r--) {
            if (rank_count[r] >= 2 && r != trip_rank) {
                result.hand_value += r;
                break;
            }
        }
        return result;
    }

    if (flush_suit >= 0) {
        result.hand_type = HAND_FLUSH;
        // Get top 5 cards of flush suit
        uint32_t val = 0;
        int count = 0;
        for (int r = 12; r >= 0 && count < 5; r--) {
            for (int i = 0; i < suit_card_count[flush_suit]; i++) {
                if (suit_cards[flush_suit][i] == r) {
                    val = val * 13 + r;
                    count++;
                    break;
                }
            }
        }
        result.hand_value = val;
        return result;
    }

    if (straight_high >= 0) {
        result.hand_type = HAND_STRAIGHT;
        result.hand_value = straight_high;
        return result;
    }

    if (trips > 0) {
        result.hand_type = HAND_THREE_KIND;
        result.hand_value = trip_rank * 169;
        // Add 2 kickers
        int kickers = 0;
        for (int r = 12; r >= 0 && kickers < 2; r--) {
            if (rank_count[r] > 0 && r != trip_rank) {
                result.hand_value += r * (kickers == 0 ? 13 : 1);
                kickers++;
            }
        }
        return result;
    }

    if (pairs >= 2) {
        result.hand_type = HAND_TWO_PAIR;
        result.hand_value = pair_ranks[0] * 169 + pair_ranks[1] * 13;
        // Add kicker
        for (int r = 12; r >= 0; r--) {
            if (rank_count[r] > 0 && r != pair_ranks[0] && r != pair_ranks[1]) {
                result.hand_value += r;
                break;
            }
        }
        return result;
    }

    if (pairs == 1) {
        result.hand_type = HAND_PAIR;
        result.hand_value = pair_ranks[0] * 2197;
        // Add 3 kickers
        int kickers = 0;
        int mult = 169;
        for (int r = 12; r >= 0 && kickers < 3; r--) {
            if (rank_count[r] > 0 && r != pair_ranks[0]) {
                result.hand_value += r * mult;
                mult /= 13;
                kickers++;
            }
        }
        return result;
    }

    // High card
    result.hand_type = HAND_HIGH_CARD;
    int count = 0;
    int mult = 28561; // 13^4
    for (int r = 12; r >= 0 && count < 5; r--) {
        if (rank_count[r] > 0) {
            result.hand_value += r * mult;
            mult /= 13;
            count++;
        }
    }
    return result;
}

// Compare two hands: returns 1 if a > b, -1 if a < b, 0 if tie
static inline int compare_hands(HandRank a, HandRank b) {
    if (a.hand_type != b.hand_type) {
        return a.hand_type > b.hand_type ? 1 : -1;
    }
    if (a.hand_value != b.hand_value) {
        return a.hand_value > b.hand_value ? 1 : -1;
    }
    return 0;
}

// ============================================================================
// Game State
// ============================================================================

typedef struct {
    // Cards
    uint8_t deck[DECK_SIZE];
    int deck_pos;
    uint8_t hole_cards[MAX_PLAYERS][HAND_SIZE];
    uint8_t community[COMMUNITY_SIZE];
    int community_count;

    // Players
    int num_players;
    float stacks[MAX_PLAYERS];
    bool active[MAX_PLAYERS];      // Still in hand
    bool all_in[MAX_PLAYERS];
    float bets[MAX_PLAYERS];       // Current round bets
    float total_bets[MAX_PLAYERS]; // Total committed this hand

    // Game state
    float pot;
    float small_blind;
    float big_blind;
    int button;                    // Dealer position
    int current_player;
    int last_raiser;
    int betting_round;             // 0=preflop, 1=flop, 2=turn, 3=river
    int raises_this_round;
    int max_raises;                // Limit poker cap
    bool hand_over;

    // For tracking
    int winner;
    float reward;

    // RNG
    uint64_t rng_state;

    // Buffers for observation/reward (set externally for zero-copy)
    float* observations;
    float* rewards;
    uint8_t* terminals;
    uint8_t* truncations;
} PokerEnv;

// ============================================================================
// Deck Operations
// ============================================================================

static void shuffle_deck(PokerEnv* env) {
    for (int i = 0; i < DECK_SIZE; i++) {
        env->deck[i] = i;
    }
    // Fisher-Yates shuffle
    for (int i = DECK_SIZE - 1; i > 0; i--) {
        int j = xorshift64(&env->rng_state) % (i + 1);
        uint8_t tmp = env->deck[i];
        env->deck[i] = env->deck[j];
        env->deck[j] = tmp;
    }
    env->deck_pos = 0;
}

static inline uint8_t deal_card(PokerEnv* env) {
    return env->deck[env->deck_pos++];
}

// ============================================================================
// Game Logic
// ============================================================================

static void init_env(PokerEnv* env, int num_players, float small_blind,
                     float big_blind, float starting_stack, uint64_t seed) {
    env->num_players = num_players;
    env->small_blind = small_blind;
    env->big_blind = big_blind;
    env->max_raises = 4; // Standard limit
    env->button = 0;
    env->rng_state = seed ? seed : 12345;

    for (int i = 0; i < num_players; i++) {
        env->stacks[i] = starting_stack;
    }

    init_hand_tables();
}

static void start_new_hand(PokerEnv* env) {
    shuffle_deck(env);

    env->pot = 0;
    env->community_count = 0;
    env->betting_round = 0;
    env->raises_this_round = 0;
    env->hand_over = false;
    env->winner = -1;

    for (int i = 0; i < env->num_players; i++) {
        env->active[i] = env->stacks[i] > 0;
        env->all_in[i] = false;
        env->bets[i] = 0;
        env->total_bets[i] = 0;
    }

    // Move button
    do {
        env->button = (env->button + 1) % env->num_players;
    } while (!env->active[env->button]);

    // Post blinds
    int sb_pos = (env->button + 1) % env->num_players;
    while (!env->active[sb_pos]) sb_pos = (sb_pos + 1) % env->num_players;

    int bb_pos = (sb_pos + 1) % env->num_players;
    while (!env->active[bb_pos]) bb_pos = (bb_pos + 1) % env->num_players;

    float sb_amount = env->small_blind < env->stacks[sb_pos] ? env->small_blind : env->stacks[sb_pos];
    float bb_amount = env->big_blind < env->stacks[bb_pos] ? env->big_blind : env->stacks[bb_pos];

    env->stacks[sb_pos] -= sb_amount;
    env->bets[sb_pos] = sb_amount;
    env->total_bets[sb_pos] = sb_amount;

    env->stacks[bb_pos] -= bb_amount;
    env->bets[bb_pos] = bb_amount;
    env->total_bets[bb_pos] = bb_amount;

    env->pot = sb_amount + bb_amount;

    // Deal hole cards
    for (int i = 0; i < env->num_players; i++) {
        if (env->active[i]) {
            env->hole_cards[i][0] = deal_card(env);
            env->hole_cards[i][1] = deal_card(env);
        }
    }

    // First to act is after BB
    env->current_player = (bb_pos + 1) % env->num_players;
    while (!env->active[env->current_player]) {
        env->current_player = (env->current_player + 1) % env->num_players;
    }
    env->last_raiser = bb_pos;
}

static float get_current_bet(PokerEnv* env) {
    float max_bet = 0;
    for (int i = 0; i < env->num_players; i++) {
        if (env->bets[i] > max_bet) max_bet = env->bets[i];
    }
    return max_bet;
}

static int count_active_players(PokerEnv* env) {
    int count = 0;
    for (int i = 0; i < env->num_players; i++) {
        if (env->active[i]) count++;
    }
    return count;
}

static int count_players_can_act(PokerEnv* env) {
    int count = 0;
    for (int i = 0; i < env->num_players; i++) {
        if (env->active[i] && !env->all_in[i]) count++;
    }
    return count;
}

static void deal_community(PokerEnv* env, int count) {
    // Burn card
    deal_card(env);
    for (int i = 0; i < count; i++) {
        env->community[env->community_count++] = deal_card(env);
    }
}

static void next_betting_round(PokerEnv* env) {
    env->betting_round++;
    env->raises_this_round = 0;

    // Reset bets
    for (int i = 0; i < env->num_players; i++) {
        env->bets[i] = 0;
    }

    // Deal community cards
    if (env->betting_round == 1) {
        deal_community(env, 3); // Flop
    } else if (env->betting_round == 2) {
        deal_community(env, 1); // Turn
    } else if (env->betting_round == 3) {
        deal_community(env, 1); // River
    }

    // First to act is after button
    env->current_player = (env->button + 1) % env->num_players;
    while (!env->active[env->current_player] || env->all_in[env->current_player]) {
        env->current_player = (env->current_player + 1) % env->num_players;
        if (env->current_player == env->button) break;
    }
    env->last_raiser = -1;
}

static void showdown(PokerEnv* env) {
    // Find best hand among active players
    HandRank best = {0, 0};
    int winners[MAX_PLAYERS];
    int num_winners = 0;

    for (int i = 0; i < env->num_players; i++) {
        if (!env->active[i]) continue;

        uint8_t cards[7];
        cards[0] = env->hole_cards[i][0];
        cards[1] = env->hole_cards[i][1];
        for (int j = 0; j < 5; j++) {
            cards[2 + j] = env->community[j];
        }

        HandRank rank = evaluate_7cards(cards);
        int cmp = compare_hands(rank, best);

        if (cmp > 0) {
            best = rank;
            winners[0] = i;
            num_winners = 1;
        } else if (cmp == 0) {
            winners[num_winners++] = i;
        }
    }

    // Split pot among winners
    float share = env->pot / num_winners;
    for (int i = 0; i < num_winners; i++) {
        env->stacks[winners[i]] += share;
    }

    env->winner = winners[0];
    env->hand_over = true;
}

// Returns: 0 = invalid action, 1 = valid
static int apply_action(PokerEnv* env, int action) {
    if (env->hand_over) return 0;

    int player = env->current_player;
    float current_bet = get_current_bet(env);
    float to_call = current_bet - env->bets[player];
    float raise_amount = env->betting_round < 2 ? env->big_blind : env->big_blind * 2;

    if (action == 0) {
        // Fold
        env->active[player] = false;

        if (count_active_players(env) == 1) {
            // Last player wins
            for (int i = 0; i < env->num_players; i++) {
                if (env->active[i]) {
                    env->stacks[i] += env->pot;
                    env->winner = i;
                    break;
                }
            }
            env->hand_over = true;
            return 1;
        }
    } else if (action == 1) {
        // Call/Check
        if (to_call > 0) {
            float call_amount = to_call < env->stacks[player] ? to_call : env->stacks[player];
            env->stacks[player] -= call_amount;
            env->bets[player] += call_amount;
            env->total_bets[player] += call_amount;
            env->pot += call_amount;

            if (env->stacks[player] == 0) {
                env->all_in[player] = true;
            }
        }
    } else if (action == 2) {
        // Raise
        if (env->raises_this_round >= env->max_raises) {
            // Can't raise, treat as call
            return apply_action(env, 1);
        }

        float raise_total = current_bet + raise_amount;
        float to_put = raise_total - env->bets[player];

        if (to_put >= env->stacks[player]) {
            // All-in
            float all_in_amount = env->stacks[player];
            env->pot += all_in_amount;
            env->bets[player] += all_in_amount;
            env->total_bets[player] += all_in_amount;
            env->stacks[player] = 0;
            env->all_in[player] = true;
        } else {
            env->stacks[player] -= to_put;
            env->bets[player] = raise_total;
            env->total_bets[player] += to_put;
            env->pot += to_put;
        }

        env->last_raiser = player;
        env->raises_this_round++;
    }

    // Find next player
    int next = (player + 1) % env->num_players;
    int checked = 0;
    while (checked < env->num_players) {
        if (env->active[next] && !env->all_in[next]) {
            break;
        }
        next = (next + 1) % env->num_players;
        checked++;
    }

    // Check if betting round is over
    bool round_over = false;
    if (count_players_can_act(env) <= 1) {
        round_over = true;
    } else if (next == env->last_raiser ||
               (env->last_raiser == -1 && next == env->current_player)) {
        round_over = true;
    } else {
        // Check if everyone has matched the bet
        float max_bet = get_current_bet(env);
        bool all_matched = true;
        for (int i = 0; i < env->num_players; i++) {
            if (env->active[i] && !env->all_in[i] && env->bets[i] < max_bet) {
                all_matched = false;
                break;
            }
        }
        if (all_matched && env->last_raiser != -1) {
            round_over = true;
        }
    }

    if (round_over) {
        if (env->betting_round == 3 || count_players_can_act(env) <= 1) {
            // Showdown or all but one all-in
            if (count_active_players(env) > 1) {
                // Deal remaining community cards if needed
                while (env->community_count < 5) {
                    deal_card(env); // Burn
                    env->community[env->community_count++] = deal_card(env);
                }
                showdown(env);
            }
        } else {
            next_betting_round(env);
            return 1;
        }
    } else {
        env->current_player = next;
    }

    return 1;
}

// ============================================================================
// Action Masking
// ============================================================================

static void get_action_mask(PokerEnv* env, uint8_t* mask) {
    // mask[0] = fold, mask[1] = call/check, mask[2] = raise
    mask[0] = 1;  // Fold siempre legal
    mask[1] = 1;  // Call/check siempre legal
    mask[2] = (env->raises_this_round < env->max_raises) ? 1 : 0;
}

// ============================================================================
// Observation Generation
// ============================================================================

static void get_observation(PokerEnv* env, int player, float* obs) {
    memset(obs, 0, OBS_SIZE * sizeof(float));
    int idx = 0;

    // Hole cards (one-hot, 52 values)
    if (env->active[player]) {
        obs[env->hole_cards[player][0]] = 1.0f;
        obs[env->hole_cards[player][1]] = 1.0f;
    }
    idx += 52;

    // Community cards (one-hot, 52 values)
    for (int i = 0; i < env->community_count; i++) {
        obs[idx + env->community[i]] = 1.0f;
    }
    idx += 52;

    // Stack sizes normalized (6 values)
    float max_stack = 0;
    for (int i = 0; i < env->num_players; i++) {
        if (env->stacks[i] > max_stack) max_stack = env->stacks[i];
    }
    if (max_stack == 0) max_stack = 1;
    for (int i = 0; i < MAX_PLAYERS; i++) {
        obs[idx++] = i < env->num_players ? env->stacks[i] / max_stack : 0;
    }

    // Pot size normalized (1 value)
    obs[idx++] = env->pot / (max_stack * env->num_players);

    // Active players (6 values)
    for (int i = 0; i < MAX_PLAYERS; i++) {
        obs[idx++] = (i < env->num_players && env->active[i]) ? 1.0f : 0;
    }

    // Betting round one-hot (4 values)
    obs[idx + env->betting_round] = 1.0f;
    idx += 4;

    // Current bets normalized (6 values)
    float max_bet = get_current_bet(env);
    if (max_bet == 0) max_bet = 1;
    for (int i = 0; i < MAX_PLAYERS; i++) {
        obs[idx++] = i < env->num_players ? env->bets[i] / max_bet : 0;
    }

    // Position relative to button (6 values)
    for (int i = 0; i < MAX_PLAYERS; i++) {
        if (i < env->num_players) {
            int rel_pos = (i - env->button + env->num_players) % env->num_players;
            obs[idx++] = (float)rel_pos / env->num_players;
        } else {
            obs[idx++] = 0;
        }
    }
}

// ============================================================================
// Batch Environment
// ============================================================================

typedef struct {
    PokerEnv* envs;
    int num_envs;
    int num_players;
    float small_blind;
    float big_blind;
    float starting_stack;

    // Shared buffers
    float* observations;
    float* rewards;
    uint8_t* terminals;
    uint8_t* truncations;
} PokerBatchEnv;

static void batch_env_init(PokerBatchEnv* batch, int num_envs, int num_players,
                          float small_blind, float big_blind, float starting_stack,
                          uint64_t seed) {
    batch->num_envs = num_envs;
    batch->num_players = num_players;
    batch->small_blind = small_blind;
    batch->big_blind = big_blind;
    batch->starting_stack = starting_stack;

    batch->envs = (PokerEnv*)calloc(num_envs, sizeof(PokerEnv));

    for (int i = 0; i < num_envs; i++) {
        init_env(&batch->envs[i], num_players, small_blind, big_blind,
                 starting_stack, seed + i);
        start_new_hand(&batch->envs[i]);
    }
}

static void batch_env_free(PokerBatchEnv* batch) {
    if (batch->envs) {
        free(batch->envs);
        batch->envs = NULL;
    }
}

static void batch_env_set_buffers(PokerBatchEnv* batch, float* obs, float* rewards,
                                  uint8_t* terminals, uint8_t* truncations) {
    batch->observations = obs;
    batch->rewards = rewards;
    batch->terminals = terminals;
    batch->truncations = truncations;
}

static void batch_env_reset(PokerBatchEnv* batch) {
    for (int i = 0; i < batch->num_envs; i++) {
        // Reset stacks
        for (int p = 0; p < batch->num_players; p++) {
            batch->envs[i].stacks[p] = batch->starting_stack;
        }
        start_new_hand(&batch->envs[i]);

        // Get observation for current player
        int player = batch->envs[i].current_player;
        get_observation(&batch->envs[i], player,
                       batch->observations + i * OBS_SIZE);
    }
}

static void batch_env_step(PokerBatchEnv* batch, int* actions, int* effective_actions) {
    for (int i = 0; i < batch->num_envs; i++) {
        PokerEnv* env = &batch->envs[i];
        int player = env->current_player;

        // Track effective action (raise might become call if capped)
        int action = actions[i];
        if (action == 2 && env->raises_this_round >= env->max_raises) {
            effective_actions[i] = 1;  // Raise capped to call
        } else {
            effective_actions[i] = action;
        }

        // Apply action
        apply_action(env, actions[i]);

        // Calculate reward for the player who acted
        float reward = 0;
        if (env->hand_over) {
            // Reward based on profit/loss
            reward = env->stacks[player] - batch->starting_stack;
            reward /= batch->starting_stack; // Normalize
        }

        batch->rewards[i] = reward;
        batch->terminals[i] = env->hand_over ? 1 : 0;
        batch->truncations[i] = 0;

        // Start new hand if needed
        if (env->hand_over) {
            // Reset stacks for new hand
            for (int p = 0; p < env->num_players; p++) {
                env->stacks[p] = batch->starting_stack;
            }
            start_new_hand(env);
        }

        // Get new observation for current player
        player = env->current_player;
        get_observation(env, player, batch->observations + i * OBS_SIZE);
    }
}

#endif // POKER_H
