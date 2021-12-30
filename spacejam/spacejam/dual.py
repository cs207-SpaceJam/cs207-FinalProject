import numpy as np


class Dual:
    """Creates dual numbers and defines dual number math.

    A real number `a` is taken in and its dual counterpart `a + eps [1.00]` is
    returned to facilitate automatic differentiation in
    :any:`spacejam.autodiff` .

    Notes
    -----
    The dual part can optionally be returned as a "dual basis vector"
    [0 1 0] if the user function `f` is multivariable and the partial
    derivative :math:`\\partial f / \\partial x_2` is desired, for example.

    Attributes
    ----------
    r : float
        real part of :any:`spacejam.dual.Dual` .
    d : numpy.ndarray
        dual part of :any:`spacejam.dual.Dual` .

    """

    def __init__(self, real, dual=None, idx=None, x=np.array(1)):
        """
        Parameters
        ----------
        real : int/float
            real part of :any:`spacejam.dual.Dual` .
        dual : float
            dual part of :any:`spacejam.dual.Dual` (default 1.00) .
        idx : int (default None)
            index in dual part of dual basis vector.
        x : numpy.ndarray (default [1])
            set size of pre-allocated array for dual basis vector.
        """
        # set numpy display output to be formatted to two decimal places
        # for easier doctesting and dedicated user dual number creation
        formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={"int_kind": formatter, "float_kind": formatter})

        # load func(p) into self.r
        self.r = np.array(real)

        # prepare self.d for differentiation
        if idx is not None:  # dual basis vector
            self.d = np.zeros(x.size)
            self.d[idx] = 1.00
        else:  # regular dual vector
            if dual is not None:
                self.d = np.array(dual)
            else:
                self.d = np.ones_like(self.r)

    def __add__(self, other):
        """Returns the addition of self and other

        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int

        Returns
        -------
        z: Dual object that is the sum of self and other

        Examples
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

    __radd__ = __add__  # addition commutes

    def __sub__(self, other):
        """Returns the subtraction of self and other

        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int

        Returns
        -------
        z: Dual object
           difference of self and other

        NOTES
        -----
        Subtraction does not commute in general.
        A specialized __rsub__ is required.

        Examples
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
        """Returns the subtraction of other from self

        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int

        Returns
        -------
        z: Dual object
           difference of other and self

        Examples
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
        """Returns the product of self and other

        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int

        Returns
        -------
        z: Dual object that is the product of self and other

        Examples
        --------
        >>> z = Dual(1, 2) * Dual(3, 4)
        >>> print(z)
        3.00 + eps 10.00
        >>> z = 2 * Dual(1, 2)
        >>> print(z)
        2.00 + eps 4.00
        """
        try:
            real = self.r * other.r
            dual = self.r * other.d + self.d * other.r
        except AttributeError:
            real = self.r * other
            dual = self.d * other

        z = Dual(real, dual)
        return z

    __rmul__ = __mul__  # multiplication commutes

    def __truediv__(self, other):
        """Returns the quotient of self and other

        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int

        Returns
        -------
        z: Dual object that is the quotient of self and other

        Examples
        --------
        >>> z = Dual(1, 2) / 2
        >>> print(z)
        0.50 + eps 1.00
        >>> z = Dual(3, 4) / Dual(1, 2)
        >>> print(z)
        3.00 - eps 2.00
        """
        try:
            real = (self.r * other.r) / other.r ** 2
            dual = (self.d * other.r - self.r * other.d) / other.r ** 2
        except AttributeError:
            real = (self.r * other) / other ** 2
            dual = (self.d * other) / other ** 2

        z = Dual(real, dual)
        return z

    def __rtruediv__(self, other):
        """Returns the quotient of other and self

        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int

        Returns
        -------
        z: Dual object that is the product of self and other

        Examples
        --------
        >>> z = 2 / Dual(1, 2)
        >>> print(z)
        2.00 - eps 4.00
        """
        real = (other * self.r) / self.r ** 2
        dual = -(other * self.d) / self.r ** 2

        z = Dual(real, dual)
        return z

    def __pow__(self, other):
        """Performs (self.r + eps self.d) ** (other.r + eps other.d)

        Parameters
        ----------
        self: Dual object
        other: Dual object, float, or int

        Returns
        -------
        z: Dual object that is self raised to the other power

        Examples
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

        if self.r < 0:
            raise Exception("The real value should be positive")

        real = self.r ** other.r
        dual = self.r ** (other.r - 1) * self.d * other.r
        if other.d != 0:
            dual += self.r ** other.r * other.d * np.log(self.r)

        z = Dual(real, dual)
        return z

    def sqrt(self):
        z = self ** (0.5)
        return z

    def __pos__(self):
        """Returns self

        Examples
        --------
        >>> z = Dual(1, 2)
        >>> print(+z)
        1.00 + eps 2.00
        """
        return Dual(self.r, self.d)

    def __neg__(self):
        """Returns negation of self

        Examples
        --------
        >>> z = Dual(1, 2)
        >>> print(-z)
        -1.00 - eps 2.00
        """
        return Dual(-self.r, -self.d)

    def exp(self):
        """Returns e**self

        Parameters
        ----------
        self: Dual object

        Returns
        -------
        z: e**self

        Examples
        --------
        >>> z = np.exp(Dual(1, 2))
        >>> print(z)
        2.72 + eps 5.44
        """
        real = np.e ** self.r
        dual = np.e ** self.r * self.d
        return Dual(real, dual)

    # Trigonometric functions that numpy looks for
    def sin(self):
        """Returns the sine of a

        Parameters
        ----------
        self: Dual object

        Returns
        -------
        z: sine of self

        Examples
        --------
        >>> z = np.sin(Dual(0, 1))
        >>> print(z)
        0.00 + eps 1.00
        """
        return Dual(np.sin(self.r), self.d * np.cos(self.r))

    def cos(self):
        """Returns the cosine of a

        Parameters
        ----------
        self: Dual object

        Returns
        -------
        z: cosine of self

        Examples
        --------
        >>> z = np.cos(Dual(0, 1))
        >>> print(z)
        1.00 + eps -0.00
        """
        return Dual(np.cos(self.r), -self.d * np.sin(self.r))

    def tan(self):
        """Returns the tangent of a

        Parameters
        ----------
        self: Dual object

        Returns
        -------
        z: tangent of self

        Examples
        --------
        >>> z = np.tan(Dual(0,1))
        >>> print(z)
        0.00 + eps 1.00
        """
        z = self.sin() / self.cos()
        return Dual(z.r, z.d)
        # return Dual(tan(self.r), self.d / np.cos(self.r)**2)

    def __repr__(self):
        """Prints self in the form a_r + eps a_d, where self = Dual(a_r, a_d),
        a_r and a_d are the real and dual part of self, respectively,
        and both terms are automatically rounded to two decimal places

        Returns
        -------
        z: Dual object that is the product of self and other

        Examples
        --------
        >>> z = Dual(1, 2)
        >>> print(z)
        1.00 + eps 2.00
        """
        # set numpy display output to be formatted to two decimal places
        # for easier doctesting
        float_formatter = lambda x: "%.2f" % x
        np.set_printoptions(
            formatter={
                "int_kind": float_formatter,
                "float_kind": float_formatter,
            }
        )

        # format dual numbers accordingly if p is scalar or vector
        if self.r.size > 1:
            s = f"{self.r} + eps {self.d}"
            return s

        if self.d.size == 1:
            if self.d >= 0:
                s = f"{self.r:.2f} + eps {self.d:.2f}"
            else:
                s = f"{self.r:.2f} - eps {np.abs(self.d):.2f}"
        else:
            s = f"{self.r:.2f} + eps {self.d}"

        return s
