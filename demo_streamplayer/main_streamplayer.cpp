#include <iostream>
#include <string>
#include "Streamplayer.h"
#include "Setting.h"

int main() {

    std::string option_path = "../option.toml";
    Setting options( option_path );
    Streamplayer player( options );
    player.start();

    return 0;
}