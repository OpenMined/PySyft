%module PyPari
%{
#include "utils.h"
%}
extern void pari_init(size_t parisize, int maxprime);
extern void pari_close();
typedef long *GEN;
extern GEN string_to_poly(char* s);
extern GEN string_to_GEN(char* s);
extern PyObject* ring_multiplication(GEN a, GEN b, GEN f, GEN q);
extern PyObject* ring_addition(GEN a, GEN b, GEN f, GEN q);

%pythoncode%{
    import atexit
    pari_init(2000000000, 2)
    atexit.register(pari_close)
%}

%include "std_string.i"

/*
%include "carrays.i"
%include "cstring.i"
%include "typemaps.i"
%include "std_vector.i"
%include "std_string.i"
%array_class(int, intArray);

namespace std {
    %template(IntVector) vector<int>;
    %template(DoubleVector) vector<double>;
    %template(StringVector) vector<char*>;
    %template(ConstCharVector) vector<const char*>;
}
*/

%pythoncode%{
    def ring_mult(a, b, f, q):
        '''
        Function for multiplication over ring R = Z_q[x]/< x^n + 1 >
        
        Args:
            a (:class:`list`): polynomial in ring R
            b (:class:`list`): polynomial in ring R
            f (:class:`list`): polynomial x^n + 1
            q (int): public modulus
        '''
        if type(a) == type(b) == type(f) == list and type(q) == int:
            a_str = b_str = f_str = ''
            for i in range(len(a)):
                a_str += str(a[i]) + ';'
            for i in range(len(b)):
                b_str += str(b[i]) + ';'
            for i in range(len(f)):
                f_str += str(f[i]) + ';'
            q_str = str(q)
            a = string_to_poly(a_str)
            b = string_to_poly(b_str)
            f = string_to_poly(f_str)
            q = string_to_GEN(q_str)
            out = ring_multiplication(a, b, f, q)
            out = out.split(';')[:-1]
            for i in range(len(out)):
                out[i] = int(out[i])
            return out
        else:
            print("Invalid Input")
            
            
    def ring_add(a, b, f, q):
        '''
        Function for multiplication over ring R = Z_q[x]/< x^n + 1 >
        
        Args:
        a (:class:`list`): polynomial in ring R
        b (:class:`list`): polynomial in ring R
        f (:class:`list`): polynomial x^n + 1
        q (int): public modulus
        '''
        if type(a) == type(b) == type(f) == list and type(q) == int:
            a_str = b_str = f_str = ''
            for i in range(len(a)):
                a_str += str(a[i]) + ';'
            for i in range(len(b)):
                b_str += str(b[i]) + ';'
            for i in range(len(f)):
                f_str += str(f[i]) + ';'
            q_str = str(q)
            a = string_to_poly(a_str)
            b = string_to_poly(b_str)
            f = string_to_poly(f_str)
            q = string_to_GEN(q_str)
            out = ring_addition(a, b, f, q)
            out = out.split(';')[:-1]
            for i in range(len(out)):
                out[i] = int(out[i])
            return out
        else:
            print("Invalid Input")
%}
