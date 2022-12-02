
#include <iostream>
#include <string>
#include "Broadcaster.h"
#include "Setting.h"


int main() {
    
    std::string option_path = "../option.toml";

    Setting options( option_path );

    Broadcaster bc( options );
    
    //等待播放端准备就绪后才开始推流，否则播放端收到的帧不完整
    std::cout << "Waiting for the player:" << std::endl;
    std::string in;
    std::cin >> in;

    bc.start();

    return 0;
}