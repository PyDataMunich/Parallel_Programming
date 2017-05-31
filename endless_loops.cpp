#include <thread>
#include <vector>

void function() {
    while(1);
}

int main() {
    std::vector<std::thread> v;
    for (int i=0; i<20; i++)
        v.emplace_back(function);
    for (auto& el: v)
        el.join();
}
