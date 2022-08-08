class LinkedInMock:
    """
    example python class for LinkedIn Banner
    """

    def __init__(self, some_var: float, is_pep8: bool) -> None:
        self.some_var = 0.0
        self.is_pep8 = True

    def brag(self) -> None:
        if self.is_pep8 is True:
            print(f"This code complies with pep8, since is_pep8 equals {self.is_pep8}.")
        elif self.some_var == 0.0:
            self.some_var += 1
        else:
            raise ValueError("something went wrong")
