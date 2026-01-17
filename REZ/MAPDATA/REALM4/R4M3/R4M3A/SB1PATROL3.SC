Name "sb1patrol3"

AbortOnDamage			FALSE		
AbortOnEnemySighted		FALSE	
CanBeMoved			FALSE	
FaceMoveDir                     TRUE
SightRange			1024
TimeOut				999
WalkSpeed			100
YawRate				180
CheckForEnemyInterval	        0.5	
TargetGroup	                "Victim"

StartAnim "AmbA"
Wait           1
StartAnim "Walk"
TurnToPos -4256 -904 -2656
MoveToPos -4256 -904 -2656

StartAnim    "Run"
WalkSpeed			240
TurnToPos -4304 -904 -2864
MoveToPos -4304 -904 -2864
TurnToTarget	0.57	
RangedAttackTarget

StartAnim "AmbA"
Wait 0.6
StartAnim "Walk"
TurnToPos -4192 -904 -2560
MoveToPos -4192 -904 -2560
TurnToPos -4272 -904 -3344
StartAnim "AmbA"

End







		
					

