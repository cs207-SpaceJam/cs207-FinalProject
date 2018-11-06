import numpy as np

class Dual():
    """ Create dual numbers and performing elementary math 
        operations.
    """

#    def __init__(self, real, dual=0):
#        """ Initializes the components of Dual object. If only real part
#            given, dual part is automatically set to zero.
#        
#        INPUTS
#        =======
#        a_r: int or float
#           real part of Dual object 
#        a_d: int or float, optional, default value is 0
#           dual part of Dual object   
#        """
#        self.r = float(real)
#        self.d = float(dual)

    def __init__(self, real, dual=0, idx=None, x=None):
        self.r = float(real)
        if idx is not None:
            self.d = np.zeros(len(x))
            self.d[idx] = 1.0
        else:
            self.d = np.array(dual)


    def __add__(self, other):
        """ Returns the addition of self and other
        
        INPUTS
        =======
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ========
        z: Dual object that is the sum of self and other
        
        EXAMPLES
        =========
        >>> z = Dual(1, 2) + Dual(3, 4)
        >>> print(z.r, z.d)
        4 6
        >>> z = 2 + Dual(1, 2)
        >>> print(z.r, z.d)
        3 2
        """
        # Recast other as Dual object if not created already
        try:
            instance_check = other.r, other.d
        except AttributeError:
            other = Dual(other, 0)

        real = self.r + other.r
        dual = self.d + other.d

        z = Dual(real, dual)
        return z
    
    __radd__ = __add__ # addition commutes
    
    def __sub__(self, other):
        """ Returns the subtraction of self and other
        
        INPUTS
        =======
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ========
        z: Dual object
           difference of self and other
           
        NOTES
        ======
        Subtraction does not commute in general. 
        A specialized __rsub__ is required. 
        
        EXAMPLES
        =========
        >>> z = Dual(1, 2) - Dual(3, 4)
        >>> print(z.r, z.d)
        -2 -2
        >>> z = Dual(1, 2) - 2
        >>> print(z.r, z.d)
        -1 2
        """
        try:
            real = self.r - other.r
            dual = self.d - other.d
        
        except AttributeError:
            real = self.r - other
            dual = self.d 
        
        z = Dual(real, dual)
        return z
    
    def __rsub__(self, other):
        """ Returns the subtraction of other from self
        
        INPUTS
        =======
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ========
        z: Dual object
           difference of other and self
        EXAMPLES
        =========
        >>> z = 2 - Dual(1, 2)
        >>> print(z.r, z.d)
        1 -2
        """
        real = other - self.r
        dual = -self.d
        
        z = Dual(real, dual)
        return z
    
    def __mul__(self, other):
        """ Returns the product of self and other
        
        INPUTS
        =======
        self: Dual object
        other: Dual object, float, or int
        RETURNS
        ========
        z: Dual object that is the product of self and other
        
        EXAMPLES
        =========
        >>> z = Dual(1, 2) * Dual(3, 4)
        >>> print(z.r, z.d)
        3 10
        >>> z = 2 * Dual(1, 2)
        >>> print(z.r, z.d)
        2 4
        """
        try:
            real = self.r*other.r 
            dual = self.r*other.d + self.d*other.r
        except AttributeError:
            real = self.r*other
            dual = self.d*other
        
        z = Dual(real, dual)
        return z

    __rmul__ = __mul__ # multiplication commutes	
    
    def __truediv__(self, other):
        """ Returns the quotient of self and other
        
        INPUTS
        =======
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ========
        z: Dual object that is the quotient of self and other
        
        EXAMPLES
        =========
        >>> z = Dual(1, 2) / 2
        >>> print(z.r, z.d)
        0.5 1.0
        >>> z = Dual(3, 4) / Dual(1, 2)
        >>> print(z.r, z.d)
        3.0 -2.0
        """
        try:
            real = (self.r*other.r)/other.r**2
            dual = (self.d*other.r - self.r*other.d)/other.r**2
        except AttributeError:
            real = (self.r*other)/other**2
            dual = (self.d*other)/other**2
        
        z = Dual(real, dual)
        return z
    
    def __rtruediv__(self, other):
        """ Returns the quotient of other and self
        
        INPUTS
        =======
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ========
        z: Dual object that is the product of self and other
        
        EXAMPLES
        =========
        >>> z = 2 / Dual(1, 2)
        >>> print(z.r, z.d)
        2.0 -4.0
        """
        real = (other*self.r)/self.r**2
        dual = -(other*self.d)/self.r**2
        
        z = Dual(real, dual)
        return z

    def __pow__(self, other):
        """ Performs (self.r + e self.d) ** (other.r + e other.d)

        INPUTS
        =======
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ========
        z: Dual object that is self raised to the other power
        
        EXAMPLES
        =========
        >>> z = Dual(1, 2) ** Dual(3, 4)
        >>> print(z.r, z.d)
        1.0 6.0
        """
        # Recast other as Dual object if not created already
        try:
            instance_check = other.r, other.d
        except AttributeError:
            other = Dual(other, 0)
       
        real = self.r**other.r
        dual = self.r**(other.r - 1)*self.d*other.r
        dual += self.r**other.r*other.d*np.log(self.r)

        z = Dual(real, dual)
        return z
            
    #Trigonometric functions
    def __sin__(a):
        """ Returns the sine of a
        
        INPUTS
        =======
        self: Dual object
        
        RETURNS
        ========
        z: sine of self
        
        EXAMPLES
        =========
        >>> z = sin(Dual(0,1))
        >>> print(z.r, z.d)
        0, 1
        """
        return Dual(sin(a.real), a.dual * cos(a.real))
        
    def __cos__(a):
        """ Returns the cosine of a
        
        INPUTS
        =======
        self: Dual object
        
        RETURNS
        ========
        z: cosine of self
        
        EXAMPLES
        =========
        >>> z = cosine(Dual(0,1))
        >>> print(z.r, z.d)
        1, 0
        """
        return Dual(cos(a.real), -a.dual * sin(a.real));

    def __tan__(a):
        """ Returns the tangent of a
        
        INPUTS
        =======
        self: Dual object
        
        RETURNS
        ========
        z: tangent of self
        
        EXAMPLES
        =========
        >>> z = tan(Dual(0,1))
        >>> print(z.r, z.d)
        0, 1
        """
    
        return Dual(tan(a.real), a.dual * sec(a.real)^2);
        
    def __repr__(self):
        """ Prints self in the form a + bi, where self = Dual(a, b),
        where a and b are automatically rounded to two decimal places
        
        RETURNS
        ========
        z: Dual object that is the product of self and other
        
        EXAMPLES
        =========
        >>> z = Dual(1, 2)
        >>> print(z)
        1 + e 2
        """
        if self.d.size == 1:
            if self.d >= 0:
                s = f'{self.r:.2f} + e {self.d:.2f}' 
            else:
                s = f'{self.r:.2f} - e {np.abs(self.d):.2f}'
        else:
            s = f'{self.r:.2f} + e {self.d}' 

        return s
