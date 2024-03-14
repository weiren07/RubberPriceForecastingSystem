def drink_price(type):
  type=type.lower()
  match type:
    case "coke":
        return 1
    case "milk":
        return 2
    case "soya":
        return 3
    case _:
        return "Unknown type. Please choose a valid option."

def get_int(note):
    while True:
        try:
            return int(input(note))
        except ValueError:
            print("Please enter a valid number.")

x = get_int("Insert notes (RM): ")
while True:
      preferred_drink = input("Insert which one you want (coke, milk, soya): ")
      price = drink_price(preferred_drink)
      if price == "Unknown type. Please choose a valid option.":
          print(price)  
      else:
          break  
number = get_int("Enter how much you want: ")
total_price=int(number)*price
if isinstance(price, int):
  remaining=x-total_price
  if remaining >= 0:
    print(f"Payment successed. Your drink is RM{total_price}. It will return you RM{remaining}.")
  else:
    print(f"Payment failed. Insufficient funds. Total price required is RM {total_price}")
else:
  print(price)
