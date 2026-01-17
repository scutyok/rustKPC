Name "ca2patrol1"

AbortOnDamage FALSE		
AbortOnEnemySighted FALSE
SightRange 768
TimeOut 999.0
FaceMoveDir TRUE
WalkSpeed 70
YawRate 180
FieldOfView 70
CheckForEnemyInterval 0.5
TargetPlayer

OnEnemyOutOfRange_Goto DISABLE
OnEnemySighted_Goto Attack				
OnDamage_Goto FarAttack			
TimeOut 10.0


Walk
StartAnim "Walk"
TurnToPos -448 0 0
MoveToPos -448 0 0
TurnToPos -672 0 672
StartAnim "Amba"




Attack:
OnEnemyOutOfRange_Goto DISABLE
OnDamage_Goto DISABLE
OnEnemySighted_Goto DISABLE
YawRate		540
TimeOut		2.5
Run
StartAnim "Run"
TurnToPos -320 0 256
MoveToPos -320 0 256
TurnToPos -640 0 768
StartAnim "Amba"
TurnToTarget	0.33	
RangedAttackTarget

FarAttack:
SightRange	1200.0



	

