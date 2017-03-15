import unittest


class testGPSpec(unittest.TestCase):

    def testIt(self):

        self.assertItemsEqual([1.0], [1.0])


if __name__ == '__main__':
    unittest.main()
