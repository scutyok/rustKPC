Name "sb1patrol5"

AbortOnDamage			FALSE		
AbortOnEnemySighted		FALSE	
CanBeMoved			FALSE	
SightRange			1024		
TimeOut				999
WalkSpeed			160
YawRate				240

StartAnim "AmbA"
Wait           2.5

StartAnim "Walk"
TurnToPos -4608 -904 -3072
StartAnim "AmbA"
Wait           1.5
YawRate				100
StartAnim "Walk"
TurnToPos -4608 -904 -3648
StartAnim "AmbA"

Wait 3

End
		
					

