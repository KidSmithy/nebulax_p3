export const CAR_PITCH = 23;

/** The camera poses and hotspot anchors are authored against Car 3; other cars are offset from it. */
export const REFERENCE_CAR = 3;

/** World-Z shift of a car (1-8) relative to where Car 3 sits. */
export const carShiftZ = (car: number) => (car - REFERENCE_CAR) * CAR_PITCH;
