Name "cm3patrol1"

AbortOnDamage FALSE
AbortOnEnemySighted FALSE
SightRange 1200
TimeOut 999.0
FaceMoveDir TRUE
YawRate 180
FieldOfView 70
WalkSpeed 70
CheckForEnemyInterval 0.5
TargetPlayer


Loop1:

OnEnemyOutOfRange_Goto DISABLE		
OnTimerExpired_Goto DISABLE	
OnEnemySighted_Goto Attack				
OnDamage_Goto FarAttack			
TimeOut 10.0

StartAnim "Amba"
Wait 1
StartAnim "Walk"
TurnTo path3
MoveTo path3
TurnToAngle 135
StartAnim "Amba"
Wait 1
StartAnim "Walk"
TurnTo path4
MoveTo path4
TurnToAngle 135

GoTo Loop1

Attack:
OnDamage_Goto DISABLE
OnEnemySighted_Goto DISABLE
OnEnemyOutOfRange_Goto	Loop1
YawRate		540
TimeOut		2.5

FarAttack:
SightRange	1600.0

if_RandomLessThan25_Goto	Attack_path4
if_RandomLessThan25_Goto	Attack_path3

Attack_path4:
TurnTo path4
Run
StartAnim	"Run"
OnTimerExpired_Goto		Attack_path4a
StartRandomTimer	0.5		0.8

MoveTo		path4

Attack_path4a:
TurnToTarget	0.33	
RangedAttackTarget

if_RandomLessThan50_Goto	Attack_path3

Attack_path3:
TurnTo path3
Run
StartAnim	"Run"
OnTimerExpired_Goto		Attack_path3a
StartRandomTimer	0.5		0.8

MoveTo		path3

Attack_path3a:
TurnToTarget	0.33	
RangedAttackTarget

if_RandomLessThan50_Goto	Attack_path4

GoTo Attack_path4




