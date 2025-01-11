from core.init import main
import scripts as user_scripts
from core.init.scripts import register_from_package, scripts

register_from_package(user_scripts, scripts)

if __name__ == "__main__":
    main()