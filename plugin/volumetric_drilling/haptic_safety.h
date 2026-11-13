#ifndef HAPTIC_SAFETY_H
#define HAPTIC_SAFETY_H

// Last step before a force goes to the Touch. On 2026-11-11 a bone contact drove
// the stylus with full-force high-frequency reversals hard enough to walk the
// base across the table. This guard makes that impossible:
//  1. Rate limit: the sent force changes by at most kMaxSlewNPerS. Normal contact
//     still reaches the device maximum in about 15 ms, but a full-amplitude
//     back-and-forth every tick cannot be sent.
//  2. Oscillation cutoff: the requested force swinging back and forth by more than
//     kSwingN, about 40 times a second or faster (no hand does that), means the
//     contact has gone unstable. The force then fades to zero and stays off
//     until the requested force has been near zero (drill out of contact) for
//     kRearmSeconds.
// Only a force request is filtered; contact, drilling and logging are unchanged.

#include <chai3d.h>
#include <algorithm>
#include <cmath>

class HapticSafety {
public:
    // A real wall: the request moves smoothly. Unstable contact: it jumps by
    // newtons and flips direction from one tick to the next.
    static constexpr double kMaxSlewNPerS = 200.0;    // 0.2 N per 1 ms tick
    static constexpr double kSwingN = 1.0;            // a swing this big, then a reversal, counts
    static constexpr double kNoiseN = 0.01;           // smaller tick changes do not set the direction
    static constexpr double kScoreDecaySeconds = 0.1;
    static constexpr double kTripScore = 8.0;         // reached by reversals every ~13 ms or faster
    static constexpr double kQuietForceN = 0.05;
    static constexpr double kRearmSeconds = 0.3;

    // dt is the physics step. Returns the force to send.
    chai3d::cVector3d filter(const chai3d::cVector3d& requested, double dt)
    {
        const double step = std::min(std::max(dt, 0.0), 0.004);

        // Oscillation score: m_swing adds up the request's change while it keeps
        // moving one way. When it turns back after a swing over kSwingN, that counts.
        const chai3d::cVector3d delta = requested - m_lastRequest;
        m_lastRequest = requested;
        if (delta.length() > kNoiseN){
            if (delta.dot(m_swing) < 0.0){
                if (m_swing.length() > kSwingN){
                    m_score += 1.0;
                }
                m_swing = delta;
            } else {
                m_swing += delta;
            }
        }
        m_score *= std::exp(-step / kScoreDecaySeconds);

        if (!m_tripped && m_score >= kTripScore){
            m_tripped = true;
            m_trips += 1;
            m_quietSeconds = 0.0;
            m_justTripped = true;
        }
        if (m_tripped){
            if (requested.length() < kQuietForceN){
                m_quietSeconds += step;
                if (m_quietSeconds >= kRearmSeconds){
                    m_tripped = false;
                    m_score = 0.0;
                }
            } else {
                m_quietSeconds = 0.0;
            }
        }

        const chai3d::cVector3d target = m_tripped ? chai3d::cVector3d(0.0, 0.0, 0.0) : requested;
        chai3d::cVector3d change = target - m_sent;
        const double maxStep = kMaxSlewNPerS * step;
        if (change.length() > maxStep){
            change.mul(maxStep / change.length());
        }
        m_sent += change;
        return m_sent;
    }

    // Pause, reset, or a force sent some other way: start again from zero.
    void reset()
    {
        m_sent.zero();
        m_lastRequest.zero();
        m_swing.zero();
        m_score = 0.0;
        m_tripped = false;
        m_quietSeconds = 0.0;
    }

    bool tripped() const { return m_tripped; }
    int trips() const { return m_trips; }
    // True once per trip, for one log line.
    bool takeJustTripped()
    {
        const bool was = m_justTripped;
        m_justTripped = false;
        return was;
    }

private:
    chai3d::cVector3d m_sent = chai3d::cVector3d(0.0, 0.0, 0.0);
    chai3d::cVector3d m_lastRequest = chai3d::cVector3d(0.0, 0.0, 0.0);
    chai3d::cVector3d m_swing = chai3d::cVector3d(0.0, 0.0, 0.0);
    double m_score = 0.0;
    bool m_tripped = false;
    bool m_justTripped = false;
    double m_quietSeconds = 0.0;
    int m_trips = 0;
};

#endif
