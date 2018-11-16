import numpy as np

class Dual():
    """ Create dual numbers and performing elementary math 
        operations.
    """

    def __init__(self, real, dual=1.00, idx=None, x=np.array(1)):
        # set numpy display output to be formatted to two decimal places
        # for easier doctesting
        formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={
                                'int_kind':formatter,
                                'float_kind':formatter
                                })

        # load func(p) into self.r
        self.r = np.array(real)

        # prepare self.d for differentiation
        if idx is not None: # dual basis vector
            self.d = np.zeros(x.size)
            self.d[idx] = 1.00
        else: # regular dual vector
            self.d = np.array(dual)

    def __add__(self, other):
        """ Returns the addition of self and other
        
        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ------- 
        z: Dual object that is the sum of self and other
        
        EXAMPLES
        -------- 
        >>> z = Dual(1, 2) + Dual(3, 4)
        >>> print(z)
        4.00 + eps 6.00
        >>> z = 2 + Dual(1, 2)
        >>> print(z)
        3.00 + eps 2.00
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
        
        parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ------- 
        z: Dual object
           difference of self and other
           
        NOTES
        ----- 
        Subtraction does not commute in general. 
        A specialized __rsub__ is required.

        EXAMPLES
        -------- 
        >>> z = Dual(1, 2) - Dual(3, 4)
        >>> print(z)
        -2.00 - eps 2.00
        >>> z = Dual(1, 2) - 2
        >>> print(z)
        -1.00 + eps 2.00
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
        
        parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ------- 
        z: Dual object
           difference of other and self

        EXAMPLES
        -------- 
        >>> z = 2 - Dual(1, 2)
        >>> print(z)
        1.00 - eps 2.00
        """
        real = other - self.r
        dual = -self.d
        
        z = Dual(real, dual)
        return z
    
    def __mul__(self, other):
        """ Returns the product of self and other
        
        parameters
        ----------
        self: Dual object
        other: Dual object, float, or int

        RETURNS
        ------- 
        z: Dual object that is the product of self and other
        
        EXAMPLES
        -------- 
        >>> z = Dual(1, 2) * Dual(3, 4)
        >>> print(z)
        3.00 + eps 10.00
        >>> z = 2 * Dual(1, 2)
        >>> print(z)
        2.00 + eps 4.00
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
        
        parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ------- 
        z: Dual object that is the quotient of self and other
        
        EXAMPLES
        -------- 
        >>> z = Dual(1, 2) / 2
        >>> print(z)
        0.50 + eps 1.00
        >>> z = Dual(3, 4) / Dual(1, 2)
        >>> print(z)
        3.00 - eps 2.00
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
        
        parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ------- 
        z: Dual object that is the product of self and other
        
        EXAMPLES
        --------  
        >>> z = 2 / Dual(1, 2)
        >>> print(z)
        2.00 - eps 4.00
        """
        real = (other*self.r)/self.r**2
        dual = -(other*self.d)/self.r**2
        
        z = Dual(real, dual)
        return z

    def __pow__(self, other):
        """ Performs (self.r + eps self.d) ** (other.r + eps other.d)

        parameters
        ----------
        self: Dual object
        other: Dual object, float, or int
        
        RETURNS
        ------- 
        z: Dual object that is self raised to the other power
        
        EXAMPLES
        -------- 
        >>> z = Dual(1, 2) ** Dual(3, 4)
        >>> print(z)
        1.00 + eps 6.00
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
            
    def __pos__(self):
        """ Returns self

        EXAMPLES
        -------- 
        >>> z = Dual(1, 2)
        >>> print(+z)
        1.00 + eps 2.00
        """
        return Dual(self.r, self.d)

    def __neg__(self):
        """ Returns negation of self

        EXAMPLES
        -------- 
        >>> z = Dual(1, 2)
        >>> print(-z)
        -1.00 - eps 2.00
        """
        return Dual(-self.r, -self.d)

    def exp(self):
        """ Returns e**self
        
        parameters
        ----------
        self: Dual object
        
        RETURNS
        ------- 
        z: e**self
        
        EXAMPLES
        -------- 
        >>> z = np.exp(Dual(1, 2))
        >>> print(z)
        2.72 + eps 5.44
        """
        real = np.e**self.r
        dual = np.e**self.r * self.d
        return Dual(real, dual)

    #Trigonometric functions that numpy looks for
    def sin(self):
        """ Returns the sine of a
        
        parameters
        ----------
        self: Dual object
        
        RETURNS
        ------- 
        z: sine of self
        
        EXAMPLES
        -------- 
        >>> z = np.sin(Dual(0, 1))
        >>> print(z)
        0.00 + eps 1.00
        """
        return Dual(np.sin(self.r), self.d*np.cos(self.r))
        
    def cos(self):
        """ Returns the cosine of a
        
        parameters
        ----------
        self: Dual object
        
        RETURNS
        ------- 
        z: cosine of self
        
        EXAMPLES
        -------- 
        >>> z = np.cos(Dual(0, 1))
        >>> print(z)
        1.00 + eps -0.00
        """
        return Dual(np.cos(self.r), -self.d*np.sin(self.r))

    def tan(self):
        """ Returns the tangent of a
        
        parameters
        ----------
        self: Dual object
        
        RETURNS
        ------- 
        z: tangent of self
        
        EXAMPLES
        -------- 
        >>> z = np.tan(Dual(0,1))
        >>> print(z)
        0.00 + eps 1.00
        """
        z = self.sin() / self.cos()
        return Dual(z.r, z.d)
        #return Dual(tan(self.r), self.d / np.cos(self.r)**2)
        
    def __repr__(self):
        """ Prints self in the form a_r + eps a_d, where self = Dual(a_r, a_d),
        a_r and a_d are the real and dual part of self, respectively,
        and both terms are automatically rounded to two decimal places
        
        RETURNS
        ------- 
        z: Dual object that is the product of self and other
        
        EXAMPLES
        -------- 
        >>> z = Dual(1, 2)
        >>> print(z)
        1.00 + eps 2.00
        """
        # set numpy display output to be formatted to two decimal places
        # for easier doctesting
        float_formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={
                                'int_kind':float_formatter,
                                'float_kind':float_formatter
                                })

        # format dual numbers accordingly if p is scalar or vector
        if self.d.size == 1:
            if self.d >= 0:
                s = f'{self.r:.2f} + eps {self.d:.2f}' 
            else:
                s = f'{self.r:.2f} - eps {np.abs(self.d):.2f}'
        else:
            s = f'{self.r:.2f} + eps {self.d}' 

        return s
