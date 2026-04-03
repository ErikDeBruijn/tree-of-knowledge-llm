"""50-prompt Ruby evaluation suite.

Import PROMPTS from this file for consistent evaluation across experiments.
Each prompt has: function signature, test code, expected output.
Covers: basic algorithms, string manipulation, array operations,
math, data structures, and Ruby-specific idioms.
"""

PROMPTS = [
    # Basic algorithms
    {"p": "def factorial(n)\n  return 1 if n <= 1\n", "t": "puts factorial(5)", "e": "120"},
    {"p": "def fibonacci(n)\n  return n if n <= 1\n", "t": "puts fibonacci(10)", "e": "55"},
    {"p": "def gcd(a, b)\n", "t": "puts gcd(12, 8)", "e": "4"},
    {"p": "def power(base, exp)\n", "t": "puts power(2, 10)", "e": "1024"},
    {"p": "def is_prime?(n)\n", "t": "puts is_prime?(7)\nputs is_prime?(4)", "e": "true\nfalse"},
    {"p": "def binary_search(arr, target)\n", "t": "puts binary_search([1, 3, 5, 7, 9], 5)", "e": "2"},
    {"p": "def lcm(a, b)\n", "t": "puts lcm(4, 6)", "e": "12"},
    {"p": "def abs_val(n)\n", "t": "puts abs_val(-5)\nputs abs_val(3)", "e": "5\n3"},
    {"p": "def clamp(n, min_val, max_val)\n", "t": "puts clamp(15, 0, 10)", "e": "10"},
    {"p": "def digit_sum(n)\n", "t": "puts digit_sum(1234)", "e": "10"},

    # String manipulation
    {"p": "def reverse_string(s)\n", "t": "puts reverse_string('hello')", "e": "olleh"},
    {"p": "def titlecase(s)\n", "t": "puts titlecase('hello world')", "e": "Hello World"},
    {"p": "def capitalize_words(s)\n", "t": "puts capitalize_words('hello world foo')", "e": "Hello World Foo"},
    {"p": "def count_vowels(s)\n", "t": "puts count_vowels('hello world')", "e": "3"},
    {"p": "def palindrome?(s)\n", "t": "puts palindrome?('racecar')\nputs palindrome?('hello')", "e": "true\nfalse"},
    {"p": "def char_frequency(s)\n", "t": "p char_frequency('hello')", "e": '{"h"=>1, "e"=>1, "l"=>2, "o"=>1}'},
    {"p": "def truncate(s, max_len)\n", "t": "puts truncate('hello world', 8)", "e": "hello..."},
    {"p": "def snake_to_camel(s)\n", "t": "puts snake_to_camel('hello_world_foo')", "e": "helloWorldFoo"},
    {"p": "def repeat_string(s, n)\n", "t": "puts repeat_string('ab', 3)", "e": "ababab"},
    {"p": "def remove_whitespace(s)\n", "t": "puts remove_whitespace('  hello  world  ')", "e": "helloworld"},

    # Array operations
    {"p": "def sum_array(arr)\n", "t": "puts sum_array([1, 2, 3, 4, 5])", "e": "15"},
    {"p": "def max_element(arr)\n", "t": "puts max_element([3, 7, 2, 9, 1])", "e": "9"},
    {"p": "def min_element(arr)\n", "t": "puts min_element([3, 7, 2, 9, 1])", "e": "1"},
    {"p": "def average(arr)\n", "t": "puts average([10, 20, 30])", "e": "20"},
    {"p": "def flatten(arr)\n", "t": "p flatten([[1, 2], [3, [4, 5]]])", "e": "[1, 2, 3, 4, 5]"},
    {"p": "def unique(arr)\n", "t": "p unique([1, 2, 2, 3, 3, 3])", "e": "[1, 2, 3]"},
    {"p": "def remove_duplicates(arr)\n", "t": "p remove_duplicates([1, 1, 2, 3, 3])", "e": "[1, 2, 3]"},
    {"p": "def intersection(a, b)\n", "t": "p intersection([1, 2, 3, 4], [3, 4, 5, 6])", "e": "[3, 4]"},
    {"p": "def rotate_array(arr, n)\n", "t": "p rotate_array([1, 2, 3, 4, 5], 2)", "e": "[3, 4, 5, 1, 2]"},
    {"p": "def zip_arrays(a, b)\n", "t": "p zip_arrays([1, 2, 3], ['a', 'b', 'c'])", "e": '[[1, "a"], [2, "b"], [3, "c"]]'},
    {"p": "def chunk_array(arr, size)\n", "t": "p chunk_array([1,2,3,4,5], 2)", "e": "[[1, 2], [3, 4], [5]]"},
    {"p": "def compact(arr)\n", "t": "p compact([1, nil, 2, nil, 3])", "e": "[1, 2, 3]"},
    {"p": "def second_largest(arr)\n", "t": "puts second_largest([3, 7, 2, 9, 1])", "e": "7"},
    {"p": "def frequencies(arr)\n", "t": "p frequencies(['a', 'b', 'a', 'c', 'b', 'a'])", "e": '{"a"=>3, "b"=>2, "c"=>1}'},
    {"p": "def take_while_positive(arr)\n", "t": "p take_while_positive([3, 1, 4, -1, 5])", "e": "[3, 1, 4]"},

    # Math
    {"p": "def even?(n)\n", "t": "puts even?(4)\nputs even?(7)", "e": "true\nfalse"},
    {"p": "def square(n)\n", "t": "puts square(7)", "e": "49"},
    {"p": "def celsius_to_fahrenheit(c)\n", "t": "puts celsius_to_fahrenheit(100)", "e": "212"},
    {"p": "def distance(x1, y1, x2, y2)\n", "t": "puts distance(0, 0, 3, 4)", "e": "5.0"},
    {"p": "def range_sum(a, b)\n", "t": "puts range_sum(1, 5)", "e": "15"},

    # Ruby-specific
    {"p": "def map_double(arr)\n", "t": "p map_double([1, 2, 3])", "e": "[2, 4, 6]"},
    {"p": "def select_even(arr)\n", "t": "p select_even([1, 2, 3, 4, 5, 6])", "e": "[2, 4, 6]"},
    {"p": "def reject_nil(arr)\n", "t": "p reject_nil([1, nil, 3, nil, 5])", "e": "[1, 3, 5]"},
    {"p": "def sum_of_squares(arr)\n", "t": "puts sum_of_squares([1, 2, 3])", "e": "14"},
    {"p": "def first_n(arr, n)\n", "t": "p first_n([10, 20, 30, 40], 2)", "e": "[10, 20]"},
    {"p": "def last_n(arr, n)\n", "t": "p last_n([10, 20, 30, 40], 2)", "e": "[30, 40]"},
    {"p": "def count_if(arr)\n  # count elements where block returns true\n", "t": "puts count_if([1,2,3,4,5]) { |x| x > 3 }", "e": "2"},
    {"p": "def to_hash(keys, values)\n", "t": "p to_hash(['a', 'b', 'c'], [1, 2, 3])", "e": '{"a"=>1, "b"=>2, "c"=>3}'},
    {"p": "def deep_copy(obj)\n", "t": "a = [1, [2, 3]]; b = deep_copy(a); b[1][0] = 99; p a[1][0]", "e": "2"},
    {"p": "def string_to_int(s)\n", "t": "puts string_to_int('42')", "e": "42"},
]

assert len(PROMPTS) == 50, f"Expected 50 prompts, got {len(PROMPTS)}"
