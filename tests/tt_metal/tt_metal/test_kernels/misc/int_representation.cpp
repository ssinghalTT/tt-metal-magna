namespace ckernel {
unsigned* instrn_buffer;
}
#include <sfpi.h>

#include "debug/dprint.h"

using namespace sfpi;

static __attribute__((noinline)) void minusOne() {
    // test loading -1 loads all bits zero

    vInt minusOne = -1;

    vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
    vUInt not2sComp = minusOne ^ allOnes;
    // not2sComp should be all bits zero
    dst_reg[0] = not2sComp;

    vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
    vUInt notSignMag = minusOne ^ signOne;
    // notSignMag will be zero, if sign-mag
    dst_reg[0] = notSignMag;
}

static __attribute__((noinline)) void signOne() {
    // test loading 0x80000001 loads as expected
    vUInt value = vUInt(1) | vUInt(0x8000) << 16;

    vUInt signOne = setsgn(vUInt(1), 1);

    vUInt notCorrect = value ^ signOne;
    // notCorrect will be zero
    dst_reg[0] = not2sComp;
    // A non-zero value
    dst_reg[0] = value;
}

static __attribute__((noinline)) void zeroMinusRegOne() {
    // test 0 - 1 in registers results in all ones.

    vInt zero = 0;
    vInt one = 1;
    vInt sub = zero - one;

    vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
    vUInt not2sComp = sub ^ allOnes;
    // not2sComp should be all bits zero
    dst_reg[0] = not2sComp;

    vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
    vUInt notSignMag = sub ^ signOne;
    // notSignMag will be zero, if sign-mag
    dst_reg[0] = notSignMag;
}

static __attribute__((noinline)) void zeroPlusRegMinusOne() {
    // test 0 + -1 in registers results in all ones.

    vInt zero = 0;
    vInt minusOne = -1;
    vInt sub = zero + minusOne;

    vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
    vUInt not2sComp = sub ^ allOnes;
    // not2sComp should be all bits zero
    dst_reg[0] = not2sComp;

    vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
    vUInt notSignMag = sub ^ signOne;
    // notSignMag will be zero, if sign-mag
    dst_reg[0] = notSignMag;
}

static __attribute__((noinline)) void zeroMinusConstOne() {
    // test 0 - 1 as cst results in all ones.

    vInt zero = 0;
    vInt sub = zero - 1;

    vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
    vUInt not2sComp = sub ^ allOnes;
    // not2sComp should be all bits zero
    dst_reg[0] = not2sComp;

    vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
    vUInt notSignMag = sub ^ signOne;
    // notSignMag will be zero, if sign-mag
    dst_reg[0] = notSignMag;
}

static __attribute__((noinline)) void zeroPlusConstMinusOne() {
    // test 0 + -1 as cst results in all ones.

    vInt zero = 0;
    vInt sub = zero + -1;

    vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
    vUInt not2sComp = sub ^ allOnes;
    // not2sComp should be all bits zero
    dst_reg[0] = not2sComp;

    vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
    vUInt notSignMag = sub ^ signOne;
    // notSignMag will be zero, if sign-mag
    dst_reg[0] = notSignMag;
}

static __attribute__((noinline)) void minusTwoASR() {
    // test -2 >> 1 is -1

    vInt minusTwo = -2;
    vInt shft = minusTwo >> 1;

    vUInt allOnes = vUInt(0xffff) | vUInt(0xffff) << 16;
    vUInt not2sComp = shft ^ allOnes;
    // not2sComp should be all bits zero
    dst_reg[0] = not2sComp;

    vUInt signOne = vUInt(1) | vUInt(0x8000) << 16;
    vUInt notSignMag = shft ^ signOne;
    // notSignMag will be zero, if sign-mag
    dst_reg[0] = notSignMag;
}

static __attribute__((noinline)) void minusTwoLSR() {
    // test unsigned(-2) >> 1 is mostPos

    vUInt minusTwo = -2;
    vUInt shft = minusTwo >> 1;

    vUInt mostPos = vUInt(0xffff) | vUInt(0x7fff) << 16;
    vUInt not2sComp = vInt(shft) ^ mostPos;
    // not2sComp should be all bits zero
    dst_reg[0] = not2sComp;

    vUInt smExpected = vUInt(1) | vUInt(0x4000) << 16;
    vUInt notSignMag = shft ^ smExpected;
    // notSignMag will be zero, if sign-mag
    dst_reg[0] = notSignMag;
}

void kernel_main() {
    unsigned test_no = *(tt_l1_ptr uint32_t*)get_arg_addr(0);
    volatile tt_l1_ptr std::uint32_t* result = (tt_l1_ptr uint32_t*)(l1_address);

    DPRINT << test_no << '\n';

    switch (test_no) {
        case 0: minusOne(); break;
        case 1: signOne(); break;
        case 2: zeroMinusRegOne(); break;
        case 3: zeroPlusRegMinusOne(); break;
        case 4: zeroMinusConstOne(); break;
        case 5: zerpPlusConstMinusOne(); break;
        case 6: minusTwoASR(); break;
        case 7: minusTwoLSR(); break;
    }
    // FIXME Copy dst_reg to l1
}
