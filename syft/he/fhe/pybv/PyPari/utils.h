#include <python.h>
#include <pari/pari.h>
#include <iostream>
#include <string>
#include <vector>

void print(GEN x){
    std::cout << GENtostr(x) << std::endl;
    return;
}

std::string poly_to_string(GEN x){
    std::string output;
    GEN temp = gtovecrev(lift(lift(x)));
    int n = (int) glength(temp);
    for(int i = 0; i < n; i++){
        output.insert(output.size(), GENtostr(gel(temp, i + 1)));
        output.push_back(';');
    }
    return output;
}

GEN string_to_poly(char* s){
    char* token = strtok(s, ";");
    std::vector<char*> components;
    while(token != NULL){
        components.push_back(token);
        token = strtok(NULL, ";");
    }
    int n = components.size();
    GEN output = cgetg(n + 1, t_VEC);
    for(int i = 0; i < n; i++)
        gel(output, i + 1) = gp_read_str(components[i]);
    components.clear();
    output = gtopolyrev(output, -1);
    return output;
}

GEN string_to_GEN(char* s){
    GEN output = gp_read_str(s);
    return output;
}

PyObject* ring_multiplication(GEN a, GEN b, GEN f, GEN q){
    a = gmodulo(a, f);
    a = gmodulo(a, q);
    b = gmodulo(b, f);
    b = gmodulo(b, q);
    GEN temp = lift(lift(gmul(a, b)));
    PyObject* output = PyString_FromString(poly_to_string(temp).c_str());
    //std::cout << output << std::endl;
    return output;
}

PyObject* ring_addition(GEN a, GEN b, GEN f, GEN q){
    a = gmodulo(a, f);
    a = gmodulo(a, q);
    b = gmodulo(b, f);
    b = gmodulo(b, q);
    GEN temp = lift(lift(gadd(a, b)));
    PyObject* output = PyString_FromString(poly_to_string(temp).c_str());
    return output;
}
