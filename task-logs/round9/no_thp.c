#include <stdio.h>
#include <sys/prctl.h>
#include <unistd.h>

int main(int argc, char ** argv) {
    if (argc < 2) {
        return 2;
    }
    if (prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0) {
        perror("PR_SET_THP_DISABLE");
        return 1;
    }
    if (prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0) != 1) {
        return 1;
    }
    execvp(argv[1], argv + 1);
    perror("execvp");
    return 1;
}
