/* Manual failure-handling checks for readFieldHeader/readFieldBody. */
#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>

int readFieldHeader(int *N, double *box_len, const char *fname, const char *dset_name);
int readFieldBody(double *box, const char *fname, const char *dset_name, int N);

static int failures = 0;

static void check(const char *what, int status, int expect_nonzero) {
    printf("%-28s status=%d (expect %s)\n", what, status,
           expect_nonzero ? "nonzero" : "0");
    if ((status != 0) != (expect_nonzero != 0)) failures++;
}

int main(void) {
    /* Build a valid 3D white-noise-like file with a Header/BoxSize attribute. */
    hid_t f = H5Fcreate("test_valid.hdf5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hid_t g = H5Gcreate(f, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    double bs[3] = {300.0, 300.0, 300.0};
    hsize_t one = 3;
    hid_t sp3 = H5Screate_simple(1, &one, NULL);
    hid_t a = H5Acreate(g, "BoxSize", H5T_NATIVE_DOUBLE, sp3, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, bs);
    H5Aclose(a); H5Sclose(sp3); H5Gclose(g);
    hid_t fg = H5Gcreate(f, "Field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[3] = {4, 4, 4};
    hid_t sp = H5Screate_simple(3, dims, NULL);
    hid_t d = H5Dcreate2(fg, "Field", H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    double data[64] = {0};
    H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(d); H5Sclose(sp); H5Gclose(fg); H5Fclose(f);

    /* A file whose dataset is 2D (malformed dimensionality). */
    f = H5Fcreate("test_malformed.hdf5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    fg = H5Gcreate(f, "Field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t d2[2] = {4, 4};
    hid_t sp2 = H5Screate_simple(2, d2, NULL);
    hid_t d2d = H5Dcreate2(fg, "Field", H5T_NATIVE_DOUBLE, sp2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dclose(d2d); H5Sclose(sp2); H5Gclose(fg); H5Fclose(f);

    int N = -1; double L = -1;

    check("missing file (header)", readFieldHeader(&N, &L, "test_missing.hdf5", "Field/Field"), 1);
    check("missing dataset (header)", readFieldHeader(&N, &L, "test_valid.hdf5", "Field/Nope"), 1);
    check("malformed dims (header)", readFieldHeader(&N, &L, "test_malformed.hdf5", "Field/Field"), 1);
    N = -1; L = -1;
    check("valid header", readFieldHeader(&N, &L, "test_valid.hdf5", "Field/Field"), 0);
    printf("  -> N=%d L=%g (expect N=4 L=300)\n", N, L);

    double *box = malloc(4 * 4 * 4 * sizeof(double));
    check("missing file (body)", readFieldBody(box, "test_missing.hdf5", "Field/Field", 4), 1);
    check("missing dataset (body)", readFieldBody(box, "test_valid.hdf5", "Field/Nope", 4), 1);
    check("valid body", readFieldBody(box, "test_valid.hdf5", "Field/Field", 4), 0);
    free(box);

    printf("%s\n", failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return failures == 0 ? 0 : 1;
}
